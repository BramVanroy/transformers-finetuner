from dataclasses import dataclass, field
from typing import Optional

from ray.tune.schedulers.pb2 import PB2


@dataclass
class OptimizationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    do_optimize: bool = field(
        default=False,
        metadata={"help": "Whether to run hyperparameter optimization"}
    )
    n_trials: int = field(
        default=8,
        metadata={"help": "Number of trials to run."}
    )
    scheduler_type: Optional[str] = field(
        default=None,
        metadata={"help": "Whether to use a scheduler, currently only supporting 'pb2'."}
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
    from ray import tune

    hp_space = {
        "num_train_epochs": tune.choice([1, 2, 3]),
    }

    scheduler = PB2(
        time_attr="training_iteration",
        metric="eval_f1",
        mode="max",
        perturbation_interval=120,
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