"""
-------------------------------------- find_misclassified.py --------------------------------
Runs the trained classifier against the validation set and saves a copy of every misclassified
image, organised by error type. This is used in conjunction with the confusion matrix.

For the current stage of the prototype, this will be helpful in identifying whether misclassified
'bad' shells (true bad, predicated good) tend to be the ones where the eyehole isn't visible in
the cropped images orientation.

Usage: python find_misclassified.py

"""

import os
import shutil
from collections import Counter

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

from model import ShellClassifier
from dataset import build_transform

def find_misclassified(val_dir="dataset_images/validation",
                       weights_path="outputs/shell_classifier.pt",
                       classes_path="outputs/classes.txt",
                       output_dir="outputs/misclassified"):

    with open(classes_path) as f:
        classes = f.read().strip().split("\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShellClassifier(img_size=128, num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = build_transform(augment=False)
    dataset = ImageFolder(val_dir, transform=transform)
    assert dataset.classes == classes, (
        f"Validation folder classes {dataset.classes} do not match "
        f"outputs/classes.txt {classes} - retrain, or check dataset_images/validation/"
        f"has the same subfolders it did when classes.txt was written"
    )

    # clean output from any previous run, so old misclassifications from
    # a prior model don't linger and get confused with the current run

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    results = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, true_idx = dataset[idx]
            path, _ = dataset.samples[idx]
            x = x.unsqueeze(0).to(device)
            probs = F.softmax(model(x), dim=1)[0]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])

            if pred_idx != true_idx:
                true_class, pred_class = classes[true_idx], classes[pred_idx]
                error_folder = os.path.join(
                    output_dir, f"true_{true_class}_predicated_{pred_class}"
                )
                os.makedirs(error_folder, exist_ok=True)

                fname = os.path.basename(path)
                # confidence prefix so files sort with models most
                # confident mistakes first
                dest = os.path.join(error_folder, f"{confidence:.0%}_{fname}")
                shutil.copyfile(path, dest)

                results.append({"path": path, "true": true_class, "predicated": pred_class, "confidence": confidence})

    print(f"Checked {len(dataset)} validation images, found {len(results)} misclassified images.\n")

    error_counts = Counter((r["true"], r["predicated"]) for r in results)
    for (true_c, pred_c), count in sorted(error_counts.items(), key=lambda kv: -kv[1]):
        print(f" {count} images: true={true_c} -> predicted={pred_c} ")

    if results:
        print(f"\nCopies saved to {output_dir}/true_<X>_predicated_<Y>/")
        print("Filenames are prefixed with the model's confidence in its"
              "(wrong) prediction - look at the highest-confidence mistakes"
              )

if __name__ == "__main__":
    find_misclassified()



