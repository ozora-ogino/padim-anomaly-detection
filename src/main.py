import argparse
import os
from pathlib import Path

from torch.utils.data import DataLoader

from dataset import MVTecDataset
from padim import PaDiM


def parse_args():
    parser = argparse.ArgumentParser("PaDiM")
    parser.add_argument("--data_path", type=Path, default="data")
    parser.add_argument("--save_path", type=Path, default="./mvtec_result")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--arch",
        type=str,
        choices=["resnet18", "wide_resnet50_2", "efficientnet"],
        default="wide_resnet50_2",
    )
    return parser.parse_args()


def _main():
    args = parse_args()
    model = PaDiM(**vars(args))

    # Train PaDiM with `good` images.
    train_dataset = MVTecDataset(args.data_path, phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    # If the model weight (distribution) is already calculated before , skip load it.
    if os.path.exists(model.train_feature_filepath):
        model.load_weight()
    else:
        model.fit(train_dataloader)

    # Evaluation.
    val_dataset = MVTecDataset(args.data_path, phase="val")
    val_dataloader = DataLoader(val_dataset, batch_size=32, pin_memory=True)
    roc_auc = model.eval(val_dataloader)
    print("Image ROCAUC: %.3f" % (roc_auc))

    # Make prediction for test data.
    test_dataset = MVTecDataset(args.data_path, phase="test")
    test_dataloader = DataLoader(test_dataset, batch_size=16, pin_memory=True)
    _ = model.predict(test_dataloader)
    print("Done!")


if __name__ == "__main__":
    _main()
