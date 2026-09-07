"""
-------------------------- monotest.py ------------------------------------
This small script allows us to quickly check the format of the images that
the physical camera has captured.
If the types change from what is expected than the code in other scripts
will need to be adjusted to handle it.

Currently, we are testing to see if the captured images are in the Mono8 format.

Usage: python monotest.py
       also update the path below to the image you want to test.

"""

import cv2

img = cv2.imread(
    r"path/to/image.png",
    cv2.IMREAD_UNCHANGED)
print("dtype:", img.dtype, "  max value:", img.max(), "  shape:", img.shape)
