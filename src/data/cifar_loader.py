import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_loader(batch_size=32, train=False, limit=500):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    # 🔥 limit dataset size
    if limit is not None:
        dataset = Subset(dataset, range(limit))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0  # important on Windows
    )

    return loader