import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_cifar10_loader(batch_size=32, train=False):

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

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )

    return loader