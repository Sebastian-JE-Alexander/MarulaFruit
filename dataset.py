"""
dataset.py
PyTorch Dataset/DataLoader for the classifier, using
torchvision.datasets.ImageFolder - it infers class labels directly from
subfolder names (dataset/train/good/, dataset/train/missing_open_eyelid/,
...), sorted alphabetically to assign indices (0=good, 1=missing_open_eyelid).
This is also what makes adding a 3rd class later a data change, not a
code change - a new subfolder is automatically picked up.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 128


def build_transform(augment: bool):
    ops = [transforms.Grayscale(num_output_channels=1)]
    if augment:
        #shells arrive at any rotation on the belt so need to transform images to mimic that
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()]
    return transforms.Compose(ops)

def make_loader(directory, augment, batch_size=16, shuffle=True):
    ds = ImageFolder(directory, transform=build_transform(augment))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, len(ds), ds.classes #ds.classes are the real class names in index order
