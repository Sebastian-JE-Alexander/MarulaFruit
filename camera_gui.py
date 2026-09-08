"""
-------------------------------- camera_gui.py ------------------------------------------
Connect up to two cameras, click trigger and see each camera's annotated result. Also
includes timing per camera and a combined time.

Cameras are identified by their configured UserDefinedName (CAM_1, CAM_2,...) rather
than based on a random enumeration order, so that the physical cameras stay consistent
across runs regardless if they get unplugged or any network errors occur. Update
CAMERA_NAMES to match the user ID's set for the cameras in MVS.

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
# apply to a plain terminal session. Adding MvImport to sys.path
# directly here means the import below works regardless of how this
# script is actually launched.
MVIMPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
if MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, MVIMPORT_DIR)

from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *

from detect_and_classify import load_model, process_frame

EXPOSURE_VAL = 172025.0  # matches cameras.py - adjust to your lighting

# Update to match the two cameras' actual configured UserDefinedName
# (set via the MVS client software).
# Add a third entry here later for the 3-camera setup;
# nothing else in this file assumes exactly two.
CAMERA_NAMES = ["CAM_1", "CAM_2", "CAM_3"]


class CameraController:
    """
    One camera connect/trigger/grab/disconnect, identified by
    UserDefinedName set in MVS.
    """

    def __init__(self, user_id, exposure=EXPOSURE_VAL):
        self.user_id = user_id
        self.cam = None
        self.exposure = exposure

    def connect(self, device_list):
        """
        device_list: an already-enumerated MV_CC_DEVICE_INFO_LIST,
        shared across all cameras being connected so enumeration only
        happens once per Connect click, not once per camera.
        """
        matched_device = None
        for i in range(device_list.nDeviceNum):
            st_device = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            name = "".join([chr(c) for c in st_device.SpecialInfo.stGigEInfo.chUserDefinedName
                            if c != 0]).strip()
            if name == self.user_id:
                matched_device = st_device
                break

        if matched_device is None:
            raise RuntimeError(f"No camera found with UserDefinedName '{self.user_id}' - "
                               f"check it's configured and powered on")

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(matched_device)
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Handle create failed, ret [0x{ret:x}]")

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Open device failed, ret [0x{ret:x}]")

        # --- Trigger config: SOFTWARE  ---
        # For the real integration, switch these two lines back to
        # hardware-trigger config instead:
        #   cam.MV_CC_SetEnumValue("TriggerSource", 0)   # Line0
        #   cam.MV_CC_SetEnumValue("TriggerActivation", 0)  # rising edge
        cam.MV_CC_SetEnumValue("TriggerMode", 1)  # 1 = trigger mode on (not free-run)
        cam.MV_CC_SetEnumValue("TriggerSource", 7)  # 7 = Software on most Hikrobot models -
        # CONFIRM against your camera's node viewer
        cam.MV_CC_SetFloatValue("ExposureTime", self.exposure)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Start grabbing failed, ret [0x{ret:x}]")

        self.cam = cam

    def grab_frame(self, timeout_ms=2000):
        """
        Fires the software trigger, retrieves one frame, returns it
        as a (H,W) uint8 numpy array - matches what process_frame()
        expects. Assumes Mono8 (1 byte/pixel).
        """
        if self.cam is None:
            raise RuntimeError(f"[{self.user_id}] Camera not connected")

        ret = self.cam.MV_CC_SetCommandValue("TriggerSoftware")
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Software trigger failed, ret [0x{ret:x}]")

        stFrame = MV_FRAME_OUT()
        memset(byref(stFrame), 0, sizeof(stFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stFrame, timeout_ms)
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Get image buffer failed, ret [0x{ret:x}]")

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
                print(f"[{self.user_id}] Error during camera cleanup: {e}")
            self.cam = None


class ShellSorterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Marula Shell Classifier - Camera Test")

        print("Loading model...")
        self.model, self.device, self.classes = load_model()
        print(f"Loaded model, classes = {self.classes}")

        self.cameras = {name: CameraController(name) for name in CAMERA_NAMES}
        self.connected = {name: False for name in CAMERA_NAMES}

        # One column per camera, side by side, each with its own image
        # and status line - so results can be compared at a glance.
        columns = tk.Frame(root)
        columns.pack(padx=10, pady=10)

        self.image_labels = {}
        self.camera_status_vars = {}
        for i, name in enumerate(CAMERA_NAMES):
            col = tk.Frame(columns, padx=10)
            col.grid(row=0, column=i)

            tk.Label(col, text=name, font=("Arial", 12, "bold")).pack()

            # NOTE: deliberately no width/height set here - Tkinter
            # measures Label width/height in CHARACTER units while
            # showing text but in PIXELS once only an image is
            # displayed, so a fixed value meant to size the text
            # placeholder collapses the box the moment an image
            # replaces the text. Auto-sizing avoids that entirely.
            img_label = tk.Label(col, text="(no image yet)")
            img_label.pack()
            self.image_labels[name] = img_label

            status_var = tk.StringVar(value="Not connected")
            tk.Label(col, textvariable=status_var, font=("Arial", 10)).pack()
            self.camera_status_vars[name] = status_var

        self.timing_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.timing_var, font=("Arial", 11, "bold")).pack(pady=(5, 0))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        self.connect_btn = tk.Button(btn_frame, text="Connect Cameras", command=self.on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        self.trigger_btn = tk.Button(btn_frame, text="Trigger", command=self.on_trigger,
                                     state=tk.DISABLED)
        self.trigger_btn.pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_connect(self):
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
            if ret != 0:
                raise RuntimeError(f"Enum Devices failed, ret [0x{ret:x}]")
            if device_list.nDeviceNum == 0:
                raise RuntimeError("No GigE cameras found - check power/network connection")
        except Exception as e:
            messagebox.showerror("Enumeration failed", str(e))
            return

        # Each camera connects independently, so one failing doesn't
        # block the other - useful when testing with only one camera
        # plugged in, or debugging which of the two has a problem.
        failures = []
        for name, cam in self.cameras.items():
            try:
                cam.connect(device_list)
                self.connected[name] = True
                self.camera_status_vars[name].set("Connected")
            except Exception as e:
                self.connected[name] = False
                self.camera_status_vars[name].set(f"FAILED: {e}")
                failures.append(name)

        if failures:
            messagebox.showwarning(
                "Partial connection",
                f"Failed to connect: {', '.join(failures)}. "
                f"Trigger will still work for whichever camera(s) connected.")

        if any(self.connected.values()):
            self.trigger_btn.config(state=tk.NORMAL)
        self.connect_btn.config(state=tk.DISABLED)

    def on_trigger(self):
        self.trigger_btn.config(state=tk.DISABLED)
        self.root.update()

        total_start = time.perf_counter()
        per_camera_ms = {}
        try:
            for name, cam in self.cameras.items():
                if not self.connected[name]:
                    continue

                capture_start = time.perf_counter()
                frame = cam.grab_frame()
                grab_ms = (time.perf_counter() - capture_start) * 1000

                annotated, results, process_ms = process_frame(
                    frame, self.model, self.device, self.classes)
                per_camera_ms[name] = grab_ms + process_ms

                self.display_frame(name, annotated)

                n_good = sum(1 for r in results if r["class"] == "good")
                n_bad = sum(1 for r in results if r["class"] == "bad")
                self.camera_status_vars[name].set(
                    f"{len(results)} shell(s) - {n_good} good, {n_bad} bad "
                    f"(capture {grab_ms:.1f} ms, processing {process_ms:.1f} ms)")

            total_ms = (time.perf_counter() - total_start) * 1000
            per_cam_summary = "  |  ".join(f"{n}: {ms:.1f} ms" for n, ms in per_camera_ms.items())
            self.timing_var.set(f"{per_cam_summary}   ||   Total (both cameras): {total_ms:.1f} ms")
        except Exception as e:
            messagebox.showerror("Trigger failed", str(e))
        finally:
            self.trigger_btn.config(state=tk.NORMAL)

    def display_frame(self, camera_name, annotated_bgr):
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # real camera frames are ~5472x3648 - shrink for on-screen display,
        # smaller per-camera since two need to fit side by side now
        pil_img.thumbnail((650, 500))
        tk_img = ImageTk.PhotoImage(pil_img)
        label = self.image_labels[camera_name]
        label.configure(image=tk_img, text="")
        label.image = tk_img  # keep a reference - Tkinter drops it otherwise

    def on_close(self):
        for cam in self.cameras.values():
            cam.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ShellSorterGUI(root)
    root.mainloop()