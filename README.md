# Marula Shell Classifier
 
A PyTorch pipeline for classifying marula shells as `good` or `bad`
from camera images, built as a proof of concept.
Segments individual shells out of a photo, classifies each one, and can run live against Hikrobot
GigE cameras through a benchtop test GUI.
 
## Requirements
 
```
pip install torch torchvision torchinfo opencv-python pillow scikit-learn matplotlib seaborn numpy
```
 
`camera_gui.py` additionally needs the Hikrobot MVS SDK's `MvImport`
folder sitting in the same directory as the script itself (see
`camera_gui.py`'s docstring - it adds `MvImport` to `sys.path`
automatically, so PyCharm's "Sources Root" setting doesn't need to be
relied on).
 
## Project structure
 
```
shell_classifier/
  config.py                     <- every shared constant lives here (see below)
  dataset_images/                <- your raw, unsegmented photos (not tracked by any script)
    train/<class>/                  raw grid photos for training
    validation/<class>/             raw grid photos for validation (physically separate shells)
  dataset/                       <- created by segment_grid_photos.py
    train/<class>/                  segmented individual shell crops
    validation/<class>/
  outputs/                       <- created by train.py / camera_gui.py
    shell_classifier.pt             trained model weights
    classes.txt                     class names, in the order the model outputs them
    training_history.png            loss/accuracy/epoch-duration plots
    confusion_matrix.png            validation confusion matrix + metrics panel
    training_log.csv                one row per training run, accumulates over time
    misclassified/                  copies of every misclassified validation image
    segmentation_check/             annotated output from segmentation_sanity_check.py
    camera_captures/                raw + annotated frames saved by camera_gui.py
    camera_results_log.csv          one row per shell detected by camera_gui.py
  MvImport/                      <- Hikrobot MVS SDK (not included here - your own files)
  *.py                           <- see Script Reference below
```
 
## Workflow
 
1. **Shoot raw photos.** 3x3 grids of shells on a plain, contrasting
   background, spread apart (not touching), same setup for every
   class. Physically separate a validation portion of your shells
   *before* photographing, and keep reshuffled/rephotographed shots of
   the same physical shells together in whichever group (train or
   validation) they started in - see "Physical-shell leakage" below.
2. **Check the raw photos, optionally.** `dataset_health_check.py` also
   works as a general dataset sanity check once step 3 has run.
3. **Segment into individual crops:**
```
   python segment_grid_photos.py --input_dir dataset_images/train/good --val_input_dir dataset_images/validation/good --class_name good
   python segment_grid_photos.py --input_dir dataset_images/train/bad --val_input_dir dataset_images/validation/bad --class_name bad
```
4. **Health-check the segmented dataset:**
```
   python dataset_health_check.py
```
5. **Train:**
```
   python train.py
```
   Review `outputs/training_history.png` and `outputs/confusion_matrix.png`.
6. **Investigate errors, if any look concerning:**
```
   python find_misclassified.py
```
7. **Test against saved images:**
```
   python detect_and_classify.py path/to/image.png
```
8. **Test against a live camera:**
```
   python camera_gui.py
```
 
## Script reference
 
| Script | Purpose |
|---|---|
| `config.py` | Every shared constant - paths, image size, segmentation thresholds, training hyperparameters, camera settings, display colours. Change a value here, not in the script using it. |
| `model.py` | The `ShellClassifier` CNN. Run directly (`python model.py`) to print an architecture summary. |
| `dataset.py` | PyTorch `Dataset`/`DataLoader` setup, including augmentation. |
| `segment_grid_photos.py` | Splits raw 3x3 grid photos into individual shell crops. Two modes: random photo-level split (`--input_dir` only) or pre-separated train/validation folders (`--input_dir` + `--val_input_dir`, required if you reshuffled photos of the same physical shells - see below). |
| `dataset_health_check.py` | Run before training: class counts/balance, corrupted or inconsistent files, unexpected colour content, exact duplicate files across train/validation. |
| `train.py` | Trains the model. Tracks loss, accuracy, and epoch duration; plots a confusion matrix with specificity/precision/recall/NPV for the binary case; logs every run to `outputs/training_log.csv`. |
| `find_misclassified.py` | Saves every misclassified validation image, grouped by error type, filename-prefixed with the model's (wrong) confidence. |
| `infer.py` | Classifies a single saved image, with timing. |
| `detect_and_classify.py` | Finds and classifies *every* shell in a saved image (not just one), draws labelled boxes, saves an annotated copy. `process_frame()` is the shared core logic also used by `camera_gui.py`. |
| `segmentation_sanity_check.py` | Visual + numeric check of segmentation quality on any folder of test frames - no dataset writes, pure diagnostic. Flags touching-shell merges and low-contrast misses. |
| `camera_gui.py` | Live benchtop GUI: connect to up to two named cameras, trigger, see annotated results side by side with timing, log every result to CSV and save every captured frame. |
 


  your specific camera model - check the MVS client's node viewer if
  triggering fails.
- PLC/actuator communication isn't built yet - `camera_gui.py`
  currently only displays results, it doesn't act on them.
 
