from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict, load_dataset
from sklearn.utils import compute_class_weight
from torch import FloatTensor
from transformers import PreTrainedTokenizer

from plot import plot_labels
from utils import float_or_int_type, logger


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "A dataset name that is available on the hub (https://huggingface.co/datasets). Preferably"
                          " with train, valid, test split. If a validation split does not exist, it will be created"
                          " automatically from the train file (stratified)."}
    )
    dataset_revision: str = field(
        default="main",
        metadata={"help": "The specific dataset version to use (can be a branch name, tag name or commit id)."},
    )

    trainsplit_name: str = field(default="train",
                                 metadata={"help": "Name of the train split in the dataset."})
    validationsplit_name: str = field(default="validation",
                                      metadata={"help": "Name of the validation split in the dataset."})
    testsplit_name: str = field(default="test", metadata={"help": "Name of the test split in the dataset."})

    validation_size: float_or_int_type = field(
        default="0.2",
        metadata={"help": "If a validation set is not present in your dataset, it will be created automatically from"
                          " the training set. You can set the ratio train/valid here (float) or an exact number of"
                          " samples that you wish to include in the validation set (int)."}
    )

    textcolumn: str = field(default="text", metadata={"help": "The column name that contains the texts."})
    labelcolumn: str = field(default="label", metadata={"help": "The column name that contains the labels."})
    split_seed: Optional[int] = field(default=None, metadata={"help": "Seed for deterministic splitting."})
    overwrite_cache: Optional[bool] = field(default=False,
                                            metadata={"help": "Whether to overwrite locally cached data files."})

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_validation_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of test examples to this "
                "value if set."
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )


@dataclass
class DataSilo(DataTrainingArguments):
    tokenizer: Optional[PreTrainedTokenizer] = None
    do_plot: bool = False
    output_dir: Path = None
    datasets: DatasetDict = field(default=None, init=False)
    regression: bool = field(default=None, init=False)

    def __post_init__(self):
        if not self.tokenizer:
            raise ValueError("A tokenizer must be given to initialize a DataSilo")

        self.train_dataset, self.test_dataset = load_dataset(self.dataset_name,
                                                             split=[self.trainsplit_name, self.testsplit_name])

        try:
            self.validation_dataset = load_dataset(self.dataset_name, split=self.validationsplit_name)
        except ValueError:
            splits = self.train_dataset.train_test_split(
                test_size=self.validation_size,
                seed=self.split_seed,
                stratify_by_column=self.labelcolumn,
                load_from_cache_file=not self.overwrite_cache)

            self.train_dataset = splits["train"]
            self.validation_dataset = splits["test"]

        if self.max_train_samples is not None:
            max_train_samples = min(len(self.train_dataset), self.max_train_samples)
            self.train_dataset = self.train_dataset.select(range(max_train_samples))

        if self.max_validation_samples is not None:
            max_validation_samples = min(len(self.validation_dataset), self.max_validation_samples)
            self.validation_dataset = self.validation_dataset.select(range(max_validation_samples))

        if self.max_test_samples is not None:
            max_test_samples = min(len(self.test_dataset), self.max_test_samples)
            self.test_dataset = self.test_dataset.select(range(max_test_samples))

        self.datasets = DatasetDict({
            "train": self.train_dataset,
            "validation": self.validation_dataset,
            "test": self.test_dataset
        })

        self._prepare_datasets()
        self.check_for_overlap_splits()

        self.labels = self.datasets["train"].features[self.labelcolumn].names
        self.label_ids = sorted(self.datasets["train"].unique(self.labelcolumn))
        self.label2id = dict(zip(self.labels, self.label_ids))
        self.id2label = dict(zip(self.label_ids, self.labels))
        self.num_labels = len(self.labels)
        self.regression = self.num_labels == 1
        self.class_weights = None

        weights = FloatTensor(compute_class_weight("balanced",
                                                   classes=self.label_ids,
                                                   y=self.datasets["train"][self.labelcolumn].numpy()))

        # If all weights are the same (all 1.) then we do not need to use weighted crossentropyloss
        # because the dataset is balanced
        if not torch.all(weights == 1.).item():
            self.class_weights = weights

        if self.do_plot:  # Plot label distributions
            plot_labels(self.datasets, self.id2label, f"{self.dataset_name} ({len(self.labels)} classes)",
                        self.output_dir)

    def _prepare_datasets(self):
        max_seq_length = min(self.max_seq_length, self.tokenizer.model_max_length) \
            if self.max_seq_length else self.tokenizer.model_max_length

        def tokenize_function(example):
            return self.tokenizer(example["text"], max_length=max_seq_length, truncation=True)

        self.datasets = self.datasets.map(tokenize_function, batched=True,
                                          desc="Tokenizing datasets",
                                          load_from_cache_file=not self.overwrite_cache)

        # Not all models/tokenizers have the same columns
        cols = [c for c in ["input_ids", "token_type_ids", "attention_mask", "label", "labels"]
                if c in self.datasets["train"].column_names]
        self.datasets.set_format(type="torch", columns=cols)

    def check_for_overlap_splits(self):
        train_text = self.train_dataset[self.textcolumn]
        valid_text = self.validation_dataset[self.textcolumn]
        test_text = self.test_dataset[self.textcolumn]
        train_valid_overlap = len(set(train_text).intersection(set(valid_text)))
        if train_valid_overlap > 0:
            logger.warn(f"Train/validation overlap: {train_valid_overlap}")

        train_test_overlap = len(set(train_text).intersection(set(test_text)))
        if train_test_overlap > 0:
            logger.warn(f"Train/test overlap: {train_test_overlap}")

        valid_test_overlap = len(set(valid_text).intersection(set(test_text)))
        if valid_test_overlap > 0:
            logger.warn(f"Validation/test overlap: {valid_test_overlap}")
