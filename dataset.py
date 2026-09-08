"""
----------------------------- dataset.py -------------------------------------------------
PyTorch Dataset/DataLoader for the classifier, using
torchvision.datasets.ImageFolder - it infers class labels directly from
subfolder names (dataset_images/train/good/, dataset_images/train/missing_open_eyelid/,
...), sorted alphabetically to assign indices (0=good, 1=missing_open_eyelid).
This is also what makes adding a 3rd class later a data change, not a
code change - a new subfolder is automatically picked up.
-----------------------------------------------------------------------------------------
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config


def build_transform(augment: bool):
    ops = [transforms.Grayscale(num_output_channels=1)]
    if augment:
        # Shells arrive at any rotation on a belt - same rationale as
        # the earlier TensorFlow and autoencoder pipelines' augmentation.
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    ops += [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()]
    return transforms.Compose(ops)


def make_loader(directory, augment, batch_size=config.BATCH_SIZE, shuffle=True):
    ds = ImageFolder(directory, transform=build_transform(augment))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, len(ds), ds.classes  # ds.classes: real class names, in index order
