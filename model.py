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
---------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn


class ShellClassifier(nn.Module):
    def __init__(self, img_size=128, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(16,32, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        flat_size = 64 * (img_size // 8) * (img_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)



if __name__=="__main__":
    from torchinfo import summary
    model = ShellClassifier(img_size=128, num_classes=2)
    summary(model, input_size=(1, 1, 128, 128))