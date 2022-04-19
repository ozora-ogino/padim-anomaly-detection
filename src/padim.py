# pylint: disable=unused-argument,too-many-function-args,undefined-loop-variable
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from random import sample
from typing import List, Optional

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import mahalanobis
from skimage import morphology
from skimage.segmentation import mark_boundaries
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision.models import resnet18, wide_resnet50_2
from tqdm import tqdm

SEED = 1024
random.seed(1024)
torch.manual_seed(1024)
use_cuda = torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.cuda.manual_seed_all(1024)


class PaDiM:
    """
    PaDiM for image anomaly detection.
    Reference: https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    """

    def __init__(self, save_path: Path, arch: str = "wide_resnet50_2", **kwargs):
        """
        Args:
            save_path (Path): Path to directory to save predict results.
            arch (str, optional): "resnet18" or "wide_resnet_50_2".
        """
        self.save_path = save_path
        self.train_feature_filepath = save_path / f"{arch}_train.pkl"
        self.img_save_path = Path(self.save_path) / f"pictures_{arch}"
        os.makedirs(self.img_save_path, exist_ok=True)

        if arch == "wide_resnet50_2":
            self.model = wide_resnet50_2(pretrained=True, progress=True)
            target_dim = 1792
            dim = 550
        elif arch == "resnet18":
            self.model = resnet18(pretrained=True, progress=True)
            target_dim = 448
            dim = 100

        self.model.to(DEVICE)
        self.model.eval()
        self.idx = torch.tensor(sample(range(0, target_dim), dim))

        # set model's intermediate outputs
        self.outputs = []

        def _hook(padim: PaDiM):
            def _func(module, _input, output):
                padim.outputs.append(output)

            return _func

        self.model.layer1[-1].register_forward_hook(_hook(self))
        self.model.layer2[-1].register_forward_hook(_hook(self))
        self.model.layer3[-1].register_forward_hook(_hook(self))

    def fit(self, dataloader, save: bool = True):
        """Learn distribution from good images."""
        train_outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])
        for (x, _, _) in tqdm(dataloader, "| feature extraction | train |"):
            with torch.no_grad():
                _ = self.model(x.to(DEVICE))
            for k, v in zip(train_outputs.keys(), self.outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []

        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs["layer1"]
        for layer_name in ["layer2", "layer3"]:
            embedding_vectors = self._embedding_concat(embedding_vectors, train_outputs[layer_name])

        # Randomly select d dimension.
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # Calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)

        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        self.train_outputs = [mean, cov]

        # Save learned distribution.
        if save:
            with open(self.train_feature_filepath, "wb") as f:
                pickle.dump(self.train_outputs, f)

    def load_weight(self, model_path: Optional[Path] = None):
        if not model_path:
            model_path = self.train_feature_filepath

        print("load train set feature from: %s" % model_path)
        with open(model_path, "rb") as f:
            self.train_outputs = pickle.load(f)

    def predict(
        self,
        dataloader,
        threshold: float = 0.5,
        dataset_name: str = "test",
        save_results: bool = True,
    ) -> np.ndarray:
        """Make prediction.

        Args:
            dataloader: Dataloader.
            threshold (float, optional): Threshold for binary classification. Defaults to 0.5.
            dataset_name (str, optional): Dataset name. This is used to save images. Defaults to "test".
            save_results (bool, optional): If true, save prediction.

        Returns:
            _type_: _description_
        """
        outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])
        test_imgs = []
        filenames_list = []
        for (x, _, filenames) in tqdm(dataloader, "| feature extraction | test |"):
            test_imgs.extend(x.cpu().detach().numpy())
            filenames_list.extend(filenames)
            with torch.no_grad():
                _ = self.model(x.to(DEVICE))

            # get intermediate layer outputs
            for k, v in zip(outputs.keys(), self.outputs):
                outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []

        scores = self._calculate_score(x, outputs)

        if save_results:
            test_save_dir = self.img_save_path / dataset_name
            os.makedirs(test_save_dir, exist_ok=True)
            save_plot_fig(test_imgs, scores, threshold, test_save_dir, filenames_list)
        return scores

    def _calculate_score(self, x: torch.Tensor, outputs: OrderedDict) -> np.ndarray:

        for k, v in outputs.items():
            outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = outputs["layer1"]
        for layer_name in ["layer2", "layer3"]:
            embedding_vectors = self._embedding_concat(embedding_vectors, outputs[layer_name])

        # randomly select d dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, self.idx)

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
        dist_list = []
        for i in range(H * W):
            mean = self.train_outputs[0][:, i]
            conv_inv = np.linalg.inv(self.train_outputs[1][:, :, i])
            dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
            dist_list.append(dist)

        dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        dist_list = torch.tensor(dist_list)
        score_map = (
            F.interpolate(
                dist_list.unsqueeze(1),
                size=x.size(2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )

        # Apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)
        return scores

    def eval(
        self,
        dataloader,
        threshold: float = 0.5,
        dataset_name: str = "val",
        save_results: bool = True,
    ) -> float:
        """Predict and evaluate.

        Args:
            Same as `predict`.

        Returns:
            float: AUC.
        """
        outputs = OrderedDict([("layer1", []), ("layer2", []), ("layer3", [])])
        gt_list = []
        test_imgs = []
        filenames_list = []

        # Extract val set features
        for (x, y, filenames) in tqdm(dataloader, "| feature extraction | val |"):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            filenames_list.extend(filenames)
            with torch.no_grad():
                _ = self.model(x.to(DEVICE))

            # get intermediate layer outputs
            for k, v in zip(outputs.keys(), self.outputs):
                outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            self.outputs = []

        scores = self._calculate_score(x, outputs)
        # Calculate image-level ROC AUC score.
        # Max value will be used as image-level score.
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)

        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        calculate_metrics(img_scores, gt_list, self.save_path / "recall-precision.png")
        img_scores = np.where(img_scores > threshold, 1, 0)
        from sklearn.metrics import accuracy_score

        print("Accuracy:", accuracy_score(img_scores, gt_list))

        # Save ROC to png.
        self._save_roc(fpr, tpr, img_roc_auc)

        if save_results:
            save_dir = self.img_save_path / dataset_name
            os.makedirs(save_dir, exist_ok=True)
            save_plot_fig(test_imgs, scores, threshold, save_dir, filenames_list)
        return img_roc_auc

    def _save_roc(self, fpr: List[float], tpr: List[float], img_roc_auc: float):
        fig = plt.figure(figsize=(20, 10))
        plt.plot(fpr, tpr, label="img_ROCAUC: %.3f" % (img_roc_auc))
        plt.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_path, "roc_curve.png"), dpi=100)
        plt.close()

    def _embedding_concat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Concatenate embedding vectors."""
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z


def calculate_metrics(pred: np.ndarray, label: np.ndarray, save_path):
    recall_list = []
    precision_list = []
    for threshold in np.arange(0.1, 1.0, 0.1):
        _pred = np.where(pred < threshold, 0, 1)
        hits = sum(_pred[_pred == label] == 1)
        # Recall
        recall = hits / sum(label == 1)
        # Presicion
        precision = hits / sum(_pred == 1)
        recall_list.append(recall)
        precision_list.append(precision)

    fig = plt.figure(figsize=(12, 3))
    plt.plot(np.arange(0.1, 1.0, 0.1), recall_list, label="recall")
    plt.plot(np.arange(0.1, 1.0, 0.1), precision_list, label="precision")
    plt.xlabel("Threshold")
    plt.legend()
    plt.tight_layout()
    fig.savefig(save_path)


def save_plot_fig(test_img, scores, threshold, save_dir, filenames_list):
    """Visualize and save predictions."""

    num_samples = len(scores)
    vmax = scores.max() * 255.0
    vmin = scores.min() * 255.0

    def _save_fig(i):
        img = test_img[i]
        img = denormalize(img)
        heat_map = scores[i] * 255
        mask = scores[i].copy()
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode="thick")
        fig_img, ax_img = plt.subplots(1, 3, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text("Image")
        ax = ax_img[1].imshow(heat_map, cmap="jet", norm=norm)
        ax_img[1].imshow(img, cmap="gray", interpolation="none")
        ax_img[1].imshow(heat_map, cmap="jet", alpha=0.5, interpolation="none")
        ax_img[1].title.set_text("Predicted heat map")
        ax_img[2].imshow(vis_img)
        ax_img[2].title.set_text("Segmentation result")

        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            "family": "serif",
            "color": "black",
            "weight": "normal",
            "size": 8,
        }
        cb.set_label("Anomaly Score", fontdict=font)

        fig_img.savefig(save_dir / filenames_list[i], dpi=100)
        plt.close()

    joblib.Parallel(n_jobs=-1, verbose=10)(joblib.delayed(_save_fig)(i) for i in range(num_samples))


def denormalize(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.0).astype(np.uint8)

    return x
