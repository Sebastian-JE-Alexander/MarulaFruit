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

# -------------------------------- Output Paths ---------------------------------
# Sets paths for all the various files that get made whilst running the scripts
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
ONNX_MODEL_PATH = f"{OUTPUTS_DIR}/shell_classifier.onnx"
ONNX_CLASSES_PATH = f"{OUTPUTS_DIR}/shell_classifier_onnx_classes.txt"

# Set explicitly if the logos/ folder ever have more than one image, and
# you need a specific one - otherwise camera_gui.py autodetects the
# first image file it finds.

LOGO_PATH = None
LOGO_DIR = "logos"


# ------------------------------------------------------------------------
# Set True to run inference via ONNX Runtime instead of PyTorch (see
# model_export.py to create the .onnx file first). ONNX_PROVIDERS lists
# execution providers in preference order - CUDAExecutionProvider needs
# onnxruntime-gpu installed AND a working CUDA/cuDNN setup; if it's not
# actually usable for any reason, ONNX Runtime silently falls back to
# the next provider in the list rather than raising an error, so
# load_model() prints which provider actually got selected at startup -
# always check that printed line rather than assuming GPU is active
# just because it was requested.
USE_ONNX = False
ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


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
CAMERA_NAMES = ["CAM_1", "CAM_2"] #adjust camera count by adding user_id set in MVS here.

# How often live_camera.py grabs+classifies a new frame - lower = more
# responsive but more CPU/GPU load. 400ms (~2.5fps) is plenty for a
# human watching a shell get placed; doesn't need to be video-smooth.
LIVE_POLL_INTERVAL_MS = 400

# ---------------------------- Display Colours --------------------------------
# BGR - OpenCV's channel order, not RGB
CLASS_COLOURS = {
    "good": (0, 200, 0),
    "bad": (0, 0, 255)
}
DEFAULT_COLOUR = (128, 0, 128)


# CLASS_NAMES is deliberately NOT hardcoded here - it's inferred from the
# actual dataset_images/train/<class>/ folder names at load time (see
# dataset.py), so it can never go out of sync with what's really on disk.


