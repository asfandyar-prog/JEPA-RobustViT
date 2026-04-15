"""
Transforms for medical imaging experiments.
Centralised here so every script uses identical preprocessing.
"""

from torchvision import transforms


# ImageNet statistics — used for all ViT-B/16 experiments
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def get_medical_transforms(train: bool = True) -> transforms.Compose:
    """
    Standard transforms for MedMNIST datasets.

    Train: resize → random flip → random crop → tensor → normalize
    Test:  resize → center crop → tensor → normalize

    Args:
        train: whether to apply training augmentations

    Returns:
        torchvision Compose transform
    """
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop(224),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


def get_tta_transforms() -> transforms.Compose:
    """
    Augmentation transforms for test-time adaptation.
    Used during TTA forward passes to estimate entropy.

    Returns:
        torchvision Compose transform
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])