"""
----------------------------- gui_theme.py --------------------------
Shared visual design tokens for every GUI script in this project
(camera_gui.py, live_camera.py, and anything else added later). One
place so multiple GUI scripts can't drift
into slightly different colours over time - same reasoning as
config.py centralizing the non-visual constants.
"""

import os
import glob

FONT_FAMILY = "Segoe UI"  # standard Windows UI font (this runs on a Windows testing
# PC); Tkinter silently falls back to a system default if
# ever unavailable, so this stays safe on other platforms too

COLOUR_BG = "#F4F5F7"  # window background
COLOUR_CARD_BG = "#FFFFFF"  # camera panel background
COLOUR_CARD_BORDER = "#E2E4E9"
COLOUR_TEXT = "#1F2430"
COLOUR_TEXT_MUTED = "#6B7280"
COLOUR_ACCENT = "#2563EB"  # primary action colour (e.g. Trigger button)

COLOUR_VERDICT = {
    "PASS": "#16A34A",
    "FAIL": "#DC2626",
    "ERROR": "#EA580C",
    "WAITING": "#9CA3AF",  # used by live_camera.py - no shell in view yet, not an error
    None: "#9CA3AF",
}


def find_logo_path(config):
    """
    Returns config.LOGO_PATH if set, and it exists, otherwise
    auto-detects the first image file in config.LOGO_DIR (your 'logos'
    folder). Returns None if nothing is found, so a GUI can skip the
    logo gracefully rather than crashing on startup. Takes config as a
    parameter rather than importing it directly, to avoid a circular
    import between this module and config.py.
    """
    if config.LOGO_PATH and os.path.isfile(config.LOGO_PATH):
        return config.LOGO_PATH
    if os.path.isdir(config.LOGO_DIR):
        candidates = sorted(
            f for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")
            for f in glob.glob(os.path.join(config.LOGO_DIR, ext))
        )
        if candidates:
            if len(candidates) > 1:
                print(f"Multiple images found in {config.LOGO_DIR}/, using "
                      f"{candidates[0]} - set config.LOGO_PATH explicitly to pick a different one.")
            return candidates[0]
    return None