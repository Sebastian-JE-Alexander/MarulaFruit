"""
-------------------------------- camera_gui.py ------------------------------------------
Camera test GUI: connect to the camera, click trigger, see the result and inference time.

Usage: python camera_gui.py
"""

import os
import sys
import time
import tkinter as tk
from tkinter import messagebox
from ctypes import *

import cv2
import numpy as np
from PIL import Image, ImageTk

# 'Sources Root' in PyCharm only affects imports when a run is launched
# through PyCharm's own Run/Debug button - it does NOT automatically
# apply to a plain terminal session (even PyCharm's integrated one,
# depending on a separate setting). Adding MvImport to sys.path
# directly here means the import below works regardless of how this
# script is actually launched.
MVIMPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
if MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, MVIMPORT_DIR)

from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *

from detect_and_classify import load_model, process_frame

EXPOSURE_VAL = 40000.0  # matches cameras.py - adjust to your lighting


class CameraController:
    """
    Single-camera connect/trigger/grab/disconnect
    """

    def __init__(self, exposure=EXPOSURE_VAL):
        self.cam = None
        self.exposure = exposure

    def connect(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
        if ret != 0:
            raise RuntimeError(f"Enum Devices failed, ret [0x{ret:x}]")
        if device_list.nDeviceNum == 0:
            raise RuntimeError("No GigE cameras found - check power/network connection")

        # First camera found - extend to loop over device_list.nDeviceNum
        # like cameras.py does once more than one camera is involved
        st_device = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(st_device)
        if ret != 0:
            raise RuntimeError(f"Handle create failed, ret [0x{ret:x}]")

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"Open device failed, ret [0x{ret:x}]")

        # --- Trigger config: SOFTWARE for this benchtop GUI ---
        # For the real belt integration, switch these two lines back to
        # hardware-trigger config instead:
        #   cam.MV_CC_SetEnumValue("TriggerSource", 0)   # Line0
        #   cam.MV_CC_SetEnumValue("TriggerActivation", 0)  # rising edge
        cam.MV_CC_SetEnumValue("TriggerMode", 1)  # 1 = trigger mode on (not free-run)
        cam.MV_CC_SetEnumValue("TriggerSource", 7)  # 7 = Software on most Hikrobot models -
        # CONFIRM against your camera's node viewer
        cam.MV_CC_SetFloatValue("ExposureTime", self.exposure)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"Start grabbing failed, ret [0x{ret:x}]")

        self.cam = cam

    def grab_frame(self, timeout_ms=2000):
        """
        Fires the software trigger, retrieves one frame, returns it
        as a (H,W) uint8 numpy array - matches what process_frame()
        expects. Assumes Mono8 (1 byte/pixel).
        """
        if self.cam is None:
            raise RuntimeError("Camera not connected")

        ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise RuntimeError(f"Software trigger failed, ret [0x{ret:x}]")

        stFrame = MV_FRAME_OUT()
        memset(byref(stFrame), 0, sizeof(stFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stFrame, timeout_ms)
        if ret != 0:
            raise RuntimeError(f"Get image buffer failed, ret [0x{ret:x}]")

        width = stFrame.stFrameInfo.nWidth
        height = stFrame.stFrameInfo.nHeight
        buf_len = stFrame.stFrameInfo.nFrameLen
        buf = (c_ubyte * buf_len)()
        memmove(byref(buf), stFrame.pBufAddr, buf_len)
        frame = np.frombuffer(buf, dtype=np.uint8, count=width * height).reshape(
            (height, width)).copy()  # .copy() - the SDK reuses pBufAddr's memory for the next frame

        self.cam.MV_CC_FreeImageBuffer(stFrame)
        return frame

    def disconnect(self):
        if self.cam:
            try:
                self.cam.MV_CC_StopGrabbing()
                self.cam.MV_CC_CloseDevice()
                self.cam.MV_CC_DestroyHandle()
            except Exception as e:
                print(f"Error during camera cleanup: {e}")
            self.cam = None


class ShellSorterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Marula Shell Classifier - Camera Test")

        print("Loading model...")
        self.model, self.device, self.classes = load_model()
        print(f"Loaded model, classes = {self.classes}")

        self.camera = CameraController()

        # NOTE: deliberately no width/height set here. Tkinter measures
        # Label width/height in CHARACTER units while showing text, but
        # in PIXELS once only an image is displayed - setting a fixed
        # width/height at creation time (meant to size the text
        # placeholder reasonably) silently collapsed the box to a tiny
        # ~60x20 PIXEL box the moment an image replaced the text,
        # regardless of the image's actual size or any thumbnail
        # resizing. Letting the Label auto-size to its actual content
        # avoids this entirely.
        self.image_label = tk.Label(root, text="(no image yet)")
        self.image_label.pack(padx=10, pady=10)

        self.status_var = tk.StringVar(value="Not connected")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 11)).pack()

        self.timing_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.timing_var, font=("Arial", 11, "bold")).pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.connect_btn = tk.Button(btn_frame, text="Connect Camera", command=self.on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        self.trigger_btn = tk.Button(btn_frame, text="Trigger", command=self.on_trigger,
                                     state=tk.DISABLED)
        self.trigger_btn.pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_connect(self):
        try:
            self.camera.connect()
            self.status_var.set("Camera connected")
            self.trigger_btn.config(state=tk.NORMAL)
            self.connect_btn.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Connection failed", str(e))
            self.status_var.set(f"Connection failed: {e}")

    def on_trigger(self):
        self.trigger_btn.config(state=tk.DISABLED)
        self.root.update()
        try:
            capture_start = time.perf_counter()
            frame = self.camera.grab_frame()
            grab_ms = (time.perf_counter() - capture_start) * 1000

            annotated, results, process_ms = process_frame(
                frame, self.model, self.device, self.classes)

            self.display_frame(annotated)

            n_good = sum(1 for r in results if r["class"] == "good")
            n_bad = sum(1 for r in results if r["class"] == "bad")
            self.status_var.set(f"Found {len(results)} shell(s) - {n_good} good, {n_bad} bad")
            self.timing_var.set(
                f"Capture: {grab_ms:.1f} ms   |   Processing: {process_ms:.1f} ms   |   "
                f"Total: {grab_ms + process_ms:.1f} ms")
        except Exception as e:
            messagebox.showerror("Trigger failed", str(e))
            self.status_var.set(f"Trigger failed: {e}")
        finally:
            self.trigger_btn.config(state=tk.NORMAL)

    def display_frame(self, annotated_bgr):
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # real camera frames are ~5472x3648 - shrink for on-screen display,
        # the full-resolution annotation quality isn't lost, just the view
        pil_img.thumbnail((900, 700))
        tk_img = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=tk_img, text="")
        self.image_label.image = tk_img  # keep a reference - Tkinter drops it otherwise

    def on_close(self):
        self.camera.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ShellSorterGUI(root)
    root.mainloop()