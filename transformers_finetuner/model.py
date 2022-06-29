from dataclasses import dataclass, field
from os import PathLike
from typing import Dict, Optional, Union

from transformers import AutoConfig, AutoModelForSequenceClassification


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )

    use_class_weights: bool = field(
        default=True,
        metadata={"help": "Whether to calculate class weights of the training set to use in CrossEntropyLoss,"
                          " to counter class imbalance."},
    )


def model_init(model_name_or_path: Union[str, PathLike], num_labels: int, label2id: Optional[Dict[str, int]] = None,
               id2label: Optional[Dict[int, str]] = None, revision: str = "main"):
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels,
                                        label2id=label2id, id2label=id2label)
    # See: https://github.com/huggingface/transformers/issues/16600
    config.num_labels = num_labels
    return AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config, revision=revision)
