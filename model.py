"""
----------------------------------- model.py -------------------------------------
CNN classifier for shell condition - good vs missing_open_eyelid to
start, can easily be extended to more classes (output layer size follows
num_classes, nothing else needs to change).

Design choices carried over from the TensorFlow

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

Note: Logits are the raw, unnormalised output values produced by the final layer
      of a neural network. They represent the model's unscaled predictions before
      they are converted into actual probabilities.
      pytorch differs here as normally you would pass the predications through
      an activation layer (like Sigmoid or Softmax in tensorflow) inside the model.
      This provides numerical stability as we are combining the activation and loss
      function into a single mathematical step to prevent issues like rounding to
      zero.

---------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn

import config


class ShellClassifier(nn.Module):
    def __init__(self, img_size=config.IMG_SIZE, num_classes=2):
        super().__init__()

        # ---------------------------- Feature Extractor ------------------------------------
        # 3 conv blocks, each halving spatial size and doubling channel count (16 -> 32 -> 64).
        # Dropout after each block increases with depth (0.2 -> 0.25 -> 0.25) since deeper
        # layers have more parameters and are more prone to overfitting on a small dataset.

        self.features = nn.Sequential(
            # Block 1: 1 input channel (grayscale) -> 16 feature maps.
            # Learns simple, low-level features (edges, basic textures)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 128 -> 64
            nn.Dropout(0.2),

            # Block 2: 16 -> 32 feature maps.
            # Combines block 1's simple features into a slightly more complex patterns.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Dropout(0.25),

            # Block 3: 32 -> 64 feature maps.
            # Highest level features this network will learn at 16x16 spatial resolution.
            # This is still fine enough to preserve small local defects, which is why
            # we use Flatten instead of pooling this further down.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Dropout(0.25),
        )

        # ------------------------ Classifier Head ----------------------------------
        # Flatten the 64x16x16 feature map into one long vector, then two dense layers
        # down to the final class scores.
        # flat_size depends on img_size since 3 MaxPool2D calls halve spatial size 3 times
        # e.g. for the default 128x128 input, this is 64 * 16 * 16 = 16384

        flat_size = 64 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.LeakyReLU(0.1, inplace=True),

            # Highest dropout rate here. This layer has the most parameters (flat_size x 128) and
            # sits right before the final decision, both reasons it's the most prone to overfitting
            # on a small dataset.
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),  # raw logits - CrossEntropyLoss applies softmax internally
        )

    def forward(self, x):
        x = self.features(x)  # extract visual features: (batch, 1, H, W) -> (batch, 64, H/8, W/8)
        return self.classifier(x) # classify those features: (batch, num_classes) raw logits.


if __name__ == "__main__":
    from torchinfo import summary

    model = ShellClassifier(img_size=config.IMG_SIZE, num_classes=2)
    summary(model, input_size=(1, config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE))
