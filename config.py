"""
-------------------------- config.py ---------------------------------------
Shared constants for the good vs missing_open_eyelid prototype.

the current code is general so that any future classes can be added
without changes.
add a third class by creating dataset_images/train/<new_class>/ and
dataset_images/validation/<new_class>/ folders - ImageFolder (used in
dataset_images.py) infers classes from folder names automatically, and the
model's output layer size is driven by len(CLASS_NAMES) here, not hardcoded.
"""

IMG_SIZE = 128
IMG_CHANNELS = 1 # native grayscale - matches our current cameras being mono8

TRAIN_DIR = "dataset_images/train"
VAL_DIR = "dataset_images/validation"

BATCH_SIZE = 16
SEED = 42

# CLASS_NAMES is set from the actual training folder names when they are loaded rather than hardcoding an amount.