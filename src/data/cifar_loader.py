import torch
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10_loader(batch_size=32, train=False, limit=500):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform
    )

    if limit is not None:
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        dataset = Subset(dataset, indices[:limit])

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )

    return loader