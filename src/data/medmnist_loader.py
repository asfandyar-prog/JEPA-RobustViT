import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO

def get_pathmnist_loader(batch_size=32, train=True):
    data_flag = "pathmnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    split = "train" if train else "test"
    dataset = DataClass(split=split, transform=transform, download=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader

def get_dermamnist_loader(batch_size=32, train=True):
    data_flag = "dermamnist"
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    split = "train" if train else "test"
    dataset = DataClass(split=split, transform=transform, download=True)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return loader
