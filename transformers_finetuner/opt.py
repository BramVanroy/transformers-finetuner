from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to optimization and early stopping.
    """
    do_optimize: bool = field(
        default=False,
        metadata={"help": "Whether to run hyperparameter optimization."}
    )
    n_trials: int = field(
        default=8,
        metadata={"help": "Number of trials to run."}
    )
    scheduler_type: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to use a scheduler, currently only supporting 'pb2'."}
    )
    do_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping."}
    )
    early_stopping_patience: int = field(
        default=1,
        metadata={"help": "If the evaluation `best metric` does not improve for this number of evaluation calls,"
                          "then stop training."}
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "The value that the model needs to improve to satisfy early stopping conditions."}
    )


def default_hp_space():
    from ray import tune

    hp_space = {
        "learning_rate": tune.loguniform(1e-4, 1e-5),  # default: 5e-5
        "gradient_accumulation_steps": tune.choice([2, 4, 8]),
        "adam_epsilon": tune.loguniform(1e-07, 1e-9),  # default: 1e-8
        "weight_decay": tune.uniform(0, 0.3),  # default (in transformers): 0.
        "adam_beta1": tune.uniform(0.85, 0.95),  # default: 0.9
        "adam_beta2": tune.uniform(0.95, 0.9999),  # default: 0.999
    }

    return hp_space


def pb2_hpspace_and_scheduler():
    from ray.tune.schedulers.pb2 import PB2
    from ray import tune

    hp_space = {
        "num_train_epochs": tune.choice([1, 2, 3, 4]),
    }

    scheduler = PB2(
        metric="eval_f1",
        mode="max",
        hyperparam_bounds={
            "weight_decay": [0.0, 0.3],  # default (in transformers): 0.
            "learning_rate": [1e-4, 1e-5],
            "gradient_accumulation_steps": [4, 8],
            "adam_epsilon": [1e-07, 1e-9],  # default: 1e-8
            "adam_beta1": [0.85, 0.9999],  # default: 0.9
            "adam_beta2": [0.95, 0.9999],  # default: 0.999
        },
    )

    return hp_space, scheduler
