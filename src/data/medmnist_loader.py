"""
MedMNIST data loaders for JEPA-RobustViT experiments.

Supported datasets:
  - PathMNIST   : 9-class colon pathology tissue (source domain)
  - DermaMNIST  : 7-class skin lesion           (shift target 1)
  - BloodMNIST  : 8-class blood cell type       (shift target 2)
  - RetinaMNIST : 5-class retinal fundus grading (shift target 3)

All datasets are resized to 224×224 and normalised with ImageNet statistics
so they are compatible with ViT-B/16 pretrained on ImageNet.
"""

import torch
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO

from src.utils.transforms import get_medical_transforms


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _build_loader(
    data_flag: str,
    split: str,
    batch_size: int,
    num_workers: int,
    train_augment: bool,
) -> DataLoader:
    """
    Generic loader builder for any MedMNIST dataset.

    Args:
        data_flag:     MedMNIST flag string e.g. 'pathmnist'
        split:         'train', 'val', or 'test'
        batch_size:    number of samples per batch
        num_workers:   DataLoader worker processes
        train_augment: whether to apply training augmentations

    Returns:
        torch DataLoader
    """
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])
    transform = get_medical_transforms(train=train_augment)

    dataset = DataClass(
        split=split,
        transform=transform,
        download=True,
        as_rgb=True,          # always 3-channel for ViT-B/16
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),   # avoids size-1 batches in training
    )
    return loader


# ---------------------------------------------------------------------------
# Public API — one function per dataset
# ---------------------------------------------------------------------------

def get_pathmnist_loader(
    batch_size: int = 64,
    split: str = "train",
    num_workers: int = 2,
) -> DataLoader:
    """
    PathMNIST: 9-class colon pathology tissue classification.
    ~100k images. This is the SOURCE domain for all domain-shift experiments.

    Args:
        batch_size:  samples per batch
        split:       'train', 'val', or 'test'
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    return _build_loader(
        data_flag="pathmnist",
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=(split == "train"),
    )


def get_dermamnist_loader(
    batch_size: int = 64,
    split: str = "test",
    num_workers: int = 2,
) -> DataLoader:
    """
    DermaMNIST: 7-class skin lesion classification.
    Domain shift target 1 — tissue → skin.

    Args:
        batch_size:  samples per batch
        split:       'train', 'val', or 'test'
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    return _build_loader(
        data_flag="dermamnist",
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=(split == "train"),
    )


def get_bloodmnist_loader(
    batch_size: int = 64,
    split: str = "test",
    num_workers: int = 4,
) -> DataLoader:
    """
    BloodMNIST: 8-class blood cell type classification.
    Domain shift target 2 — tissue → blood cells.

    Args:
        batch_size:  samples per batch
        split:       'train', 'val', or 'test'
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    return _build_loader(
        data_flag="bloodmnist",
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=(split == "train"),
    )


def get_retinamnist_loader(
    batch_size: int = 64,
    split: str = "test",
    num_workers: int = 2,
) -> DataLoader:
    """
    RetinaMNIST: 5-class retinal fundus grading.
    Domain shift target 3 — tissue → retinal fundus.

    Args:
        batch_size:  samples per batch
        split:       'train', 'val', or 'test'
        num_workers: DataLoader workers

    Returns:
        DataLoader
    """
    return _build_loader(
        data_flag="retinamnist",
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        train_augment=(split == "train"),
    )


# ---------------------------------------------------------------------------
# Dataset metadata — used by training scripts
# ---------------------------------------------------------------------------

DATASET_NUM_CLASSES = {
    "pathmnist":  9,
    "dermamnist": 7,
    "bloodmnist": 8,
    "retinamnist": 5,
}

DATASET_LOADERS = {
    "pathmnist":  get_pathmnist_loader,
    "dermamnist": get_dermamnist_loader,
    "bloodmnist": get_bloodmnist_loader,
    "retinamnist": get_retinamnist_loader,
}