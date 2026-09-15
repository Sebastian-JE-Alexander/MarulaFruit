"""
----------------------------- dataset.py -------------------------------------------------
PyTorch Dataset/DataLoader for the classifier, using
torchvision.datasets.ImageFolder - it infers class labels directly from
subfolder names (dataset_images/train/good/, dataset_images/train/bad/,
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
    # Grayscale conversion happens here regardless of the file that was loaded
    # This is more of a defensive measure than a redundant feature. If a file
    # was added that had colour content (RGB), this forces it back to single
    # channel rather than giving the model 3 channels that it wasn't built for.
    # dataset_health_check.py performs a colour check to catch this upstream.
    ops = [transforms.Grayscale(num_output_channels=1)]

    if augment:
        # Shells arrive at any rotation on a belt.
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),

            # degrees=180, allows for a random angle in range [-180, +180]
            # this allows for a full 360 degree spread of orientations.
            transforms.RandomRotation(degrees=180),

            # Real shells vary in lighting/cleanliness (dirt, tarnish)
            # Mild brightness/contrast jitter teaches the model not to
            # rely on exact pixel brightness as a shortcut.
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]

    # Resize + ToTensor always comes last, after augmentation, not before.
    # Augmenting on the original resolution preserves fine detail through
    # the geometric transforms, and ToTensor must be the final step since it
    # converts from a PIL image to a tensor, after which none of the PIL-based
    # transforms above could run any more.
    ops += [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)), transforms.ToTensor()]
    return transforms.Compose(ops)


def make_loader(directory, augment, batch_size=config.BATCH_SIZE, shuffle=True):

    # ImageFolder infers the class labels from subfolder names automatically.
    # ds.classes lists them in the same alphabetical order used to assign
    # indices (0=first, 1=second, ...), this is the same order used by the model's
    # output logits.
    # Returning this alongside the loader means callers never need to hardcode class
    # names/order, see load_model() in detect_and_classify.py
    ds = ImageFolder(directory, transform=build_transform(augment))

    # num_workers=0 single process data loading.
    # Slower than using multiple worker processes for large datasets, but avoids
    # common error on pytorch where multiprocessing Dataloader workers need
    # extra guarding.
    # If dataset expands greatly then will need to implement it.
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader, len(ds), ds.classes  # ds.classes: real class names, in index order
