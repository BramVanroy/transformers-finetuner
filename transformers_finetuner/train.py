import os
import sys
from dataclasses import asdict
from functools import partial
from json import dump
from pathlib import Path
from random import randint

import numpy as np
import torch
from ray.tune import CLIReporter

from transformers import (AutoTokenizer,
                          EarlyStoppingCallback, HfArgumentParser, Trainer,
                          TrainingArguments,
                          )

from transformers_finetuner.data import DataSilo, DataTrainingArguments
from transformers_finetuner.env import Env
from transformers_finetuner.model import ModelArguments, model_init
from transformers_finetuner.opt import OptimizationArguments, default_hp_space, pb2_hpspace_and_scheduler
from transformers_finetuner.plot import plot_confusion_matrix
from transformers_finetuner.trainer import WeightedTrainer, compute_metrics, compute_objective
from transformers_finetuner.utils import change_logger_level, get_last_ckpt, logger, merge_and_save_dataclasses


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, OptimizationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        margs, dargs, targs, oargs = parser.parse_json_file(json_file=Path(sys.argv[1]).resolve())
    else:
        margs, dargs, targs, oargs = parser.parse_args_into_dataclasses()

    # Commenting for now: seems to be bugged when reloading checkpoints
    # targs.optim = "adamw_torch"  # Explicitly set to torch AdamW rather than HF AdamW
    targs.ray_scope = "all"  # pick the best checkpoint from all chkpts of trials rather than just the last

    output_dir = Path(targs.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if targs.seed is None:
        targs.seed = randint(0, 1024)
        logger.warning(f"No 'seed' was set. New random seed {dargs.split_seed} chosen.")

    if dargs.split_seed is None:
        dargs.split_seed = randint(0, 1024)
        logger.warning(f"No 'split_seed' was set for dataset splitting, so IF automatic splitting needs to occur, we"
                       f" will likely not be able to use a cached dataset . New random seed {dargs.split_seed} chosen.")

    change_logger_level(targs.get_process_log_level())

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {targs.local_rank}, device: {targs.device}, n_gpu: {targs.n_gpu}"
        + f"distributed training: {bool(targs.local_rank != -1)}, 16-bits training: {targs.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = get_last_ckpt(output_dir, targs.resume_from_checkpoint,
                                    targs.do_train, targs.overwrite_output_dir)

    # Init datasets into a datasilo
    distrib = int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False
    tokenizer = AutoTokenizer.from_pretrained(margs.model_name_or_path)
    datasilo = DataSilo(**asdict(dargs), tokenizer=tokenizer, output_dir=output_dir, do_plot=False,
                        is_distributed=distrib, is_world_process_zero=targs.process_index)

    # Tie our current parameters to model_init, so that it can be invoked at each hparam search
    if datasilo.regression:
        _model_init = partial(model_init, margs.model_name_or_path, 1, None, None, margs.model_revision)
    else:
        _model_init = partial(model_init, margs.model_name_or_path, datasilo.num_labels, datasilo.label2id,
                              datasilo.id2label, margs.model_revision)

    targs.metric_for_best_model = "r2" if datasilo.regression else "f1"
    targs.greater_is_better = True
    targs.load_best_model_at_end = True

    # Select a class to use as trainer. Custom trainer for classification to use class weights
    trainer_class = (partial(WeightedTrainer, class_weights=datasilo.class_weights)
                     if datasilo.class_weights is not None and not datasilo.regression and margs.use_class_weights else Trainer)

    # Prepare a computer_metrics based on whether or not we use regression
    _compute_metrics = partial(compute_metrics, regression=datasilo.regression, label_ids=None, predictions=None,
                               calculate_qwk=oargs.calculate_qwk)

    early_stopper = EarlyStoppingCallback(early_stopping_patience=oargs.early_stopping_patience,
                                          early_stopping_threshold=oargs.early_stopping_threshold)
    trainer: Trainer = trainer_class(
        model_init=_model_init,
        args=targs,
        tokenizer=tokenizer,
        train_dataset=datasilo.datasets["train"],
        eval_dataset=datasilo.datasets["validation"],
        compute_metrics=_compute_metrics,
        callbacks=[early_stopper] if oargs.do_early_stopping else None
    )

    if oargs.do_optimize:
        # Do not raise an exception on failed experiments. These may occur due to OOM when tuning batch_size
        # raise_on_failed_trial=False
        device_count = torch.cuda.device_count()
        resources_per_trial = None
        if device_count > 0:
            resources_per_trial = {"cpu": min(4, (os.cpu_count() - 1) // device_count),
                                   "gpu": 1}

        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "wd",
                "learning_rate": "lr",
                "gradient_accumulation_steps": "accumul",
                "adam_epsilon": "eps",
                "adam_beta1": "beta1",  # default: 0.9
                "adam_beta2": "beta2"
            },
            metric_columns=["eval_f1", "eval_loss", "steps"],
        )

        scheduler = None
        hpspace = None
        if oargs.scheduler_type:
            if oargs.scheduler_type == "pb2":
                hpspace, scheduler = pb2_hpspace_and_scheduler()
        else:
            hpspace = default_hp_space()

        best_params = trainer.hyperparameter_search(hp_space=lambda _: hpspace,
                                                    backend="ray",
                                                    n_trials=oargs.n_trials,
                                                    raise_on_failed_trial=False,
                                                    resources_per_trial=resources_per_trial,
                                                    keep_checkpoints_num=1,
                                                    scheduler=scheduler,
                                                    compute_objective=compute_objective,
                                                    progress_reporter=reporter,
                                                    local_dir=output_dir.joinpath("ray_results"),
                                                    log_to_file=True)

        if not oargs.scheduler_type:
            # Set the trainer to the best hyperparameters found
            for hparam, v in best_params.hyperparameters.items():
                setattr(trainer.args, hparam, v)

            if trainer.is_world_process_zero():
                with output_dir.joinpath("opt_hparams.json").open("w", encoding="utf-8") as hp_out:
                    dump(best_params, hp_out, indent=4, sort_keys=True)

    if targs.do_train:
        checkpoint = None
        if targs.resume_from_checkpoint is not None:
            checkpoint = targs.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if trainer.is_world_process_zero():
            metrics = train_result.metrics
            trainer.save_model()
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    if targs.do_predict and datasilo.datasets["test"] is not None:
        logger.info("*** Predict/test ***")

        gold_labels = datasilo.datasets["test"]["label"].numpy()
        predict_dataset = datasilo.datasets["test"].remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions

        if trainer.is_world_process_zero():
            predictions = predictions.squeeze() if datasilo.regression else np.argmax(predictions, axis=1)

            preds_df = datasilo.datasets["test"].to_pandas()
            preds_df["preds"] = predictions

            if not datasilo.regression:
                preds_df["pred_labels"] = [datasilo.id2label[pred] for pred in predictions]

            results = compute_metrics(label_ids=gold_labels, predictions=predictions, regression=datasilo.regression,
                                      calculate_qwk=oargs.calculate_qwk)
            if datasilo.regression:  # TODO: check if this is a mistake?
                results = {**compute_metrics(label_ids=gold_labels, predictions=predictions, regression=False),
                           **results}

            plot_confusion_matrix(predictions, gold_labels, datasilo.labels, output_dir)
            trainer.log_metrics("test", results)
            trainer.save_metrics("test", results)

            preds_df = preds_df.drop(columns=["input_ids", "token_type_ids", "attention_mask"], errors="ignore")
            preds_df.to_csv(output_dir.joinpath(f"predictions_test.txt"), index=False, sep="\t")

    if trainer.is_world_process_zero():
        # Save args and env to output dir
        Env.dump(output_dir.joinpath("env.json"))
        merge_and_save_dataclasses(margs, dargs, targs, oargs, output_file=output_dir.joinpath("args.json"))


if __name__ == "__main__":
    main()
