"""
----------------------------------- model.py -------------------------------------
CNN classifier for shell condition - good vs missing_open_eyelid to
start, extendable to more classes later (output layer size follows
num_classes, nothing else needs to change).

Design choices carried over from the TensorFlow,
where they were found to matter for this kind of small, localized
visual signal:
  * Flatten before the dense layers, NOT GlobalAveragePooling2D/AvgPool.
    GAP averages every spatial location together, which destroys the
    signal from a defect that only occupies a handful of pixels (the
    eyelid hole, a hairline crack).

  * No BatchNorm. With a small dataset and augmentation, BatchNorm's
    running statistics mismatched between augmented training batches and
    clean inference input and collapsed predictions in an earlier test.
    Dropout + L2-equivalent (weight_decay in the optimizer, see train.py)
    handle regularization instead.

  * Grayscale input (1 channel) - the physical camera is monochrome (Mono8),
    and neither class needs colour/hue to distinguish for the moment.

  * LeakyReLU instead of plain ReLU. A real training run collapsed
    completely (loss stuck at ln(2), every prediction the same class)
    from an unlucky random initialization - plain ReLU can permanently
    "die" (output exactly zero, and therefore have exactly zero
    gradient forever after) if enough units get pushed negative early
    on, especially with a small dataset offering little signal to
    recover with. LeakyReLU lets a small gradient through even for
    negative inputs, so a unit that starts in a bad spot can still
    receive a learning signal and correct itself, rather than getting
    stuck permanently.
---------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn

import config


class ShellClassifier(nn.Module):
    def __init__(self, img_size=config.IMG_SIZE, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Dropout(0.25),
        )

        flat_size = 64 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),  # raw logits - CrossEntropyLoss applies softmax internally
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    from torchinfo import summary

    model = ShellClassifier(img_size=config.IMG_SIZE, num_classes=2)
    summary(model, input_size=(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
