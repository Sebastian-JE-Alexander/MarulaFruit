"""
-------------------------- config.py ---------------------------------------

Single source of for constants used across the project.

Importing from here means changing a value once actually changes it everywhere.
-------------------------------------------------------------------------------
"""

# ----------------------------- Image/Model ------------------------------------------
IMG_SIZE = 128
IMG_CHANNELS = 1 # native grayscale - matches our current cameras being mono8

# -------------------------------- Dataset paths ---------------------------------
DATASET_DIR = "dataset_images"
TRAIN_DIR = f"{DATASET_DIR}/train"
VAL_DIR = f"{DATASET_DIR}/validation"

# -------------------------------- Output Paths ---------------------------
OUTPUTS_DIR = "outputs"
MODEL_WEIGHTS_PATH = f"{OUTPUTS_DIR}/shell_classifier.pt"
CLASSES_PATH = f"{OUTPUTS_DIR}/classes.txt"
TRAINING_LOG_PATH = f"{OUTPUTS_DIR}/training_log.csv"
TRAINING_DURATION_PATH = f"{OUTPUTS_DIR}/training_duration.txt"
TRAINING_HISTORY_PLOT_PATH = f"{OUTPUTS_DIR}/training_history.png"
CONFUSION_MATRIX_PLOT_PATH = f"{OUTPUTS_DIR}/confusion_matrix.png"
MISCLASSIFIED_DIR = f"{OUTPUTS_DIR}/misclassified"
SEGMENTATION_CHECK_DIR = f"{OUTPUTS_DIR}/segmentation_check"
CAMERA_RESULTS_LOG_PATH = f"{OUTPUTS_DIR}/camera_results_log.csv"
CAMERA_CAPTURES_DIR = f"{OUTPUTS_DIR}/camera_captures"

# Set explicitly if the logos/ folder ever have more than one image and
# you need a specific one - otherwise camera_gui.py autodetects the
# first image file it finds.

LOGO_PATH = None
LOGO_DIR = "logos"


# ---------------------------- Segmentation -------------------------------------
# Tuned for 5472x3648 pixel camera frames -rescale if camera resolution or distance
# changes meaningfully
MIN_BLOB_AREA = 5000
MAX_BLOB_ASPECT = 2.5
EXPECTED_SHELLS_PER_GRID_PHOTO = 9

# ------------------------- Training -------------------------------------------
BATCH_SIZE = 16
SEED = 42
DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 8
VAL_FRACTION = 0.2 # for segment_grid_photos.py random-split mode

# ------------------------------ Camera ----------------------------------------
EXPOSURE_VAL = 172025.0
CAMERA_NAMES = ["CAM_1", "CAM_2", "CAM_3"]

# ---------------------------- Display Colours --------------------------------
# BGR - OpenCV's channel order, not RGB
CLASS_COLOURS = {
    "good": (0, 200, 0),
    "bad": (0, 0, 255)
}
DEFAULT_COLOR = (128, 0, 128)


# CLASS_NAMES is deliberately NOT hardcoded here - it's inferred from the
# actual dataset_images/train/<class>/ folder names at load time (see
# dataset.py), so it can never go out of sync with what's really on disk.


