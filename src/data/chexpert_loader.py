import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):

        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        self.df = self.df.fillna(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_root, row["Path"])

        image = Image.open(img_path).convert("RGB")

        labels = torch.tensor(row[LABELS].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        return image, labels


def get_chexpert_loader(
        data_dir,
        batch_size=32,
        train=True):

    csv_file = "train.csv" if train else "valid.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CheXpertDataset(
        csv_path=os.path.join(data_dir, csv_file),
        img_root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4
    )

    return loader