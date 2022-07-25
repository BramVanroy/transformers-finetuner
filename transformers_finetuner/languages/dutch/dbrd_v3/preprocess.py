import logging
from io import TextIOWrapper
from pathlib import Path
import sys
from shutil import unpack_archive
from time import sleep
from typing import Literal

import requests
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)

ARCHIVE_URL = "https://github.com/benjaminvdb/DBRD/releases/download/v3.0/DBRD_v3.tgz"

"""Adapted and extended from https://github.com/iPieter/RobBERT/blob/master/src/preprocess_dbrd.py"""


def download_and_extract_archive():
    """Download and unpack the archive in the directory of this script. Skip steps if the data already exists."""
    arch_fname = Path(ARCHIVE_URL).name
    arch_path = Path(__file__).parent.joinpath(arch_fname)
    response = requests.get(ARCHIVE_URL, stream=True)

    if not arch_path.with_suffix("").exists():
        if not arch_path.exists():
            with tqdm.wrapattr(open(arch_path, "wb"), "write",
                               miniters=1, desc=arch_fname,
                               total=int(response.headers.get("content-length", 0))) as fhout:
                for chunk in response.iter_content(chunk_size=4096):
                    fhout.write(chunk)
                fhout.flush()  # Make sure all the data is in there

            # Sleep to make sure that the whole file is written and flushed
            # Otherwise may encounter an issue with corrupt data
            sleep(2)

        logging.info(f"Extracting {arch_fname}. May take a while...")
        unpack_archive(arch_path, arch_path.with_suffix(""))
    else:
        logging.info(f"{arch_fname} already downloaded and extracted!")

    return arch_path.with_suffix("").resolve()


def add_content_and_label(pfin: Path, fhout: TextIOWrapper, label: int):
    """Read the content from a file, turn it into a single line of text (no newlines)
    and add the label at the end (tab-separated). Then write this all to the given file handle."""
    content = " ".join(pfin.read_text(encoding="utf-8").splitlines(keepends=True))
    content = content.replace('\n', ' ').replace('\r', '')  # replace content by one single line
    single_spaced_content = " ".join(content.split())  # make sure that we have no duplicate spaces
    fhout.write(f"{single_spaced_content}\t{label}\n")


def process_dbrd(corpus_path: Path, test_or_train: Literal["train", "test"]):
    """Use the data from the extracted archive to generate compatible data for our own pipeline. So:
    1. Open an output pipe to an output file (train.txt or test.txt);
    2. Collect all positive and negative files and sort them (determinism). Every line is a small text;
    3. Convert the small texts into single-line strings and write them to the output file with their appropriate
    label (0 - negative, postive - 1)"""
    dout = Path(__file__).parent.joinpath(f"{corpus_path.stem}_processed")
    corpus_path = corpus_path / "DBRD"
    dout.mkdir(parents=True, exist_ok=True)

    pfout = dout / f"{test_or_train}.txt"

    def get_file_id(pfin: Path):
        return int(pfin.stem.split("_")[0])

    logging.info(f"Processing {test_or_train} files...")
    with pfout.open(mode="w", encoding="utf-8") as fhout:
        pos_files_folder = corpus_path / test_or_train / 'pos'
        neg_files_folder = corpus_path / test_or_train / 'neg'

        pos_files = sorted([p for p in pos_files_folder.glob("*") if p.is_file()], key=get_file_id)
        neg_files = sorted([p for p in neg_files_folder.glob("*") if p.is_file()], key=get_file_id)

        if len(pos_files) != len(neg_files):
            raise ValueError("Expected the same number of positive and negative files. This error might indicate that"
                             " your dataset was incomplete.")

        # process file by intertwining the files, such that the model can learn better
        # Note BV: this is not _really_ necessary since we'll shuffle anyway
        # Keeping it so that we can make train/dev/test splits in the same way as RobBERT
        for i in range(len(pos_files)):
            add_content_and_label(pos_files_folder / pos_files[i].name, fhout, 1)
            add_content_and_label(neg_files_folder / neg_files[i].name, fhout, 0)

    return pfout


def split_train_into_traindev(train_p: Path):
    """Split the train file into a train and validation set. The first 500 will be
    for validation, the rest for training.
    This is a Python simulation of what was done in RobBERT in shell
    See: https://github.com/iPieter/RobBERT/blob/master/src/split_dbrd_training.sh
    """
    logging.info("Splitting train set into train and validation sets...")
    valid_p = train_p.parent / train_p.name.replace("train", "validation")
    lines = train_p.read_text(encoding="utf-8").splitlines(keepends=True)

    valid_p.write_text("".join(lines[:500]), encoding="utf-8")
    train_p.write_text("".join(lines[500:]), encoding="utf-8")

    return train_p, valid_p


def verify_files(train_p, valid_p, test_p):
    train_text = train_p.read_text(encoding="utf-8").splitlines()
    # valid_text = valid_p.read_text(encoding="utf-8").splitlines()
    test_text = test_p.read_text(encoding="utf-8").splitlines()

    all_good = True

    # train_valid_overlap = set(train_text).intersection(set(valid_text))
    # if len(train_valid_overlap) > 0:
    #     logging.warning(f"Train/validation overlap: {len(train_valid_overlap)}")
    #     print(train_valid_overlap)
    #     all_good = False

    train_test_overlap = set(train_text).intersection(set(test_text))
    if len(train_test_overlap) > 0:
        logging.warning(f"Train/test overlap: {len(train_test_overlap)}")
        print(train_test_overlap)
        all_good = False
    #
    # valid_test_overlap = set(valid_text).intersection(set(test_text))
    # if len(valid_test_overlap) > 0:
    #     logging.warning(f"Validation/test overlap: {len(valid_test_overlap)}")
    #     print(valid_test_overlap)
    #     all_good = False

    if all_good:
        logging.info("All splits verified. No overlap!")


def main():
    corpus_path = download_and_extract_archive()
    train_p = process_dbrd(corpus_path, "train")
    test_p = process_dbrd(corpus_path, "test")
    # train_p, valid_p = split_train_into_traindev(train_p)
    verify_files(train_p, None, test_p)

if __name__ == "__main__":
    main()
