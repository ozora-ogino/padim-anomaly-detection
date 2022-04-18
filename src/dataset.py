import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MVTecDataset(Dataset):
    def __init__(
        self,
        dataset_path: str = "./data",
        phase: str = "train",
        resize: int = 256,
        cropsize: int = 224,
    ):
        self.dataset_path = Path(dataset_path)
        self.phase = phase
        self.resize = resize
        self.cropsize = cropsize

        # load dataset
        self.x, self.y, self.filenames = self.load_dataset_folder()

        # set transforms
        self.transform_x = T.Compose(
            [
                T.Resize(resize, Image.ANTIALIAS),
                T.CenterCrop(cropsize),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST), T.CenterCrop(cropsize), T.ToTensor()])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, List[int], List[str]]:
        x, y, filenames = self.x[idx], self.y[idx], self.filenames[idx]

        x = Image.open(x).convert("RGB")
        x = self.transform_x(x)

        return x, y, filenames

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self) -> Tuple[List[str], List[int], List[str]]:
        x, y, filenames = [], [], []

        img_dir = self.dataset_path / self.phase

        if self.phase == "test":
            img_fpath_list = sorted([img_dir / f for f in os.listdir(img_dir) if f.endswith(".png")])
            filenames.extend([os.path.basename(x) for x in img_fpath_list])
            return img_fpath_list, [-1] * len(img_fpath_list), filenames

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = img_dir / img_type

            # Skip files. (e.g. LICENSE)
            if not os.path.isdir(img_type_dir):
                continue

            img_fpath_list = sorted([img_type_dir / f for f in os.listdir(img_type_dir) if f.endswith(".png")])
            x.extend(img_fpath_list)
            filenames.extend([os.path.basename(x) for x in img_fpath_list])

            # If good, add 0 to y.
            if img_type == "good":
                y.extend([0] * len(img_fpath_list))

            else:
                y.extend([1] * len(img_fpath_list))
        assert len(x) == len(y), "number of x and y should be same"
        return list(x), list(y), filenames
