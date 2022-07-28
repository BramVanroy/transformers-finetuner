import dataclasses
import logging
import sys
from argparse import ArgumentTypeError
from json import dump
from os import PathLike, listdir
from pathlib import Path
from typing import Union, Dict
import hashlib

import datasets
import transformers
from transformers.trainer_utils import get_last_checkpoint


def init_logger():
    log = logging.getLogger(__name__)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log.setLevel("INFO")

    return log


logger = init_logger()


def change_logger_level(log_level: str = "INFO"):
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def get_last_ckpt(output_dir: Union[str, PathLike], resume_from_checkpoint: Union[str, PathLike],
                  do_train: bool, overwrite_output_dir: bool):
    last_checkpoint = None
    if Path(output_dir).is_dir() and do_train and not overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is None and len(listdir(output_dir)) > 0:
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    return last_checkpoint


def float_or_int_type(arg: str) -> Union[float, int]:
    """Allows for een argument parser to accept both a float or int.
    Note that this is very naive, and won't work for scientific notations - only
    with dotted floats, e.g. 4.2
    :param arg: input item (float or int)
    :return: the intified or floatified item
    """
    likely_float = "." in arg
    try:
        arg = float(arg)
    except ValueError:
        raise ArgumentTypeError(f"{arg} is not a float-able input")

    if not likely_float:
        arg = int(arg)

    return arg


def merge_and_save_dataclasses(*dcs, output_file: Union[str, PathLike]) -> Dict:
    args = {}
    for d in dcs:
        args.update(dataclasses.asdict(d))

    with Path(output_file).open("w", encoding="utf-8") as args_out:
        dump(args, args_out, indent=4, sort_keys=True)

    return args


def create_hash_from_str(text: str, length: int = 64) -> str:
    # Using a length of 64 as that is the max char length that datasets accepts
    return str(int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**length)
