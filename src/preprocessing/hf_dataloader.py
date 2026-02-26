import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]
        label = self.data.iloc[idx]["label"]

        image = Image.open(img_path).convert("RGB")

        return image, label


def collate_fn(batch, processor):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])

    inputs = processor(images=images, return_tensors="pt")
    return inputs["pixel_values"], labels


def get_dataloaders(train_csv, val_csv, processor, batch_size=32):

    train_dataset = CSVDataset(train_csv)
    val_dataset = CSVDataset(val_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: collate_fn(x, processor)
    )

    return train_loader, val_loader