import os
import random
import shutil
from argparse import ArgumentParser
from pathlib import Path


def train_test_split(path: Path, ratio: float = 0.8):
    """Split train and val."""
    good_files = os.listdir(path)
    random.shuffle(good_files)
    num_train_sample = int(len(good_files) * ratio)
    train_good_files, val_good_files = good_files[:num_train_sample], good_files[num_train_sample:]
    return train_good_files, val_good_files


def _main(archive_path: str, data_path: str):
    """
    Create directory for this project.
    This project uses PaDiM which needs only good cases to fit.
    So good case will be splitted for validation and all anomalies will be used in validation.

    Args:
        archive_path (str): Path to your MVTec Data.
    """
    ARCHIVE_DIR = Path(archive_path)
    DATA_DIR = Path(data_path)
    TRAIN_ARCHIVE_DIR = ARCHIVE_DIR / "train"
    TEST_ARCHIVE_DIR = ARCHIVE_DIR / "test"
    TRAIN_DATA_DIR = DATA_DIR / "train"
    VAL_DATA_DIR = DATA_DIR / "val"
    TEST_DATA_DIR = DATA_DIR / "test"
    GOOD_ARCHIVE_DIR = TRAIN_ARCHIVE_DIR / "good"
    ANOMALY_ARCHIVE_DIR = TRAIN_ARCHIVE_DIR / "not-good"

    train_good_files, val_good_files = train_test_split(GOOD_ARCHIVE_DIR)

    os.makedirs(VAL_DATA_DIR / "good", exist_ok=True)
    os.makedirs(TRAIN_DATA_DIR / "good", exist_ok=True)

    for file in train_good_files:
        src = GOOD_ARCHIVE_DIR / file
        dest = TRAIN_DATA_DIR / "good" / file
        shutil.copyfile(src, dest)

    for file in val_good_files:
        src = GOOD_ARCHIVE_DIR / file
        dest = VAL_DATA_DIR / "good" / file
        shutil.copyfile(src, dest)

    src = ANOMALY_ARCHIVE_DIR
    dest = VAL_DATA_DIR / "not-good"

    shutil.copytree(src, dest)
    shutil.copytree(TEST_ARCHIVE_DIR, TEST_DATA_DIR)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--archive_path", default="archive")
    parser.add_argument("--data_path", default="data")
    args = parser.parse_args()
    _main(**vars(args))
