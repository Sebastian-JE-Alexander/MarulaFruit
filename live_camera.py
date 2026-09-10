"""
live_camera.py
Live demo view: continuously streams frames from both cameras and runs
segmentation + classification on every frame.

IMPORTANT DIFFERENCE FROM camera_gui.py: that script configures the
camera for SOFTWARE TRIGGERING (TriggerMode=1) because a GUI button
click IS the trigger there - one grab per click. This script needs the
OPPOSITE: FREE-RUN / CONTINUOUS acquisition (TriggerMode=0), so the
camera just keeps streaming frames on its own and this script
continuously pulls whatever's latest. Confirm TriggerMode's "off/free-
run" value against your camera's MVS client node viewer the same way
TriggerSource needed confirming for camera_gui.py's software-trigger path.

Reuses the exact same model/segmentation/classification/fusion pipeline
as camera_gui.py (load_model, process_frame, fuse_pass_fail) - only the
camera acquisition mode and the display loop differ. HOW a frame gets
judged good/bad is identical between the two scripts, by design - so a
result seen here means the same thing it would in camera_gui.py.


Usage: python live_camera.py
"""

import os
import sys
import time
import tkinter as tk
from ctypes import *

import cv2
import numpy as np
from PIL import Image, ImageTk

MVIMPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport")
if MVIMPORT_DIR not in sys.path:
    sys.path.insert(0, MVIMPORT_DIR)

from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *

from detect_and_classify import load_model, process_frame, fuse_pass_fail
from gui_theme import (FONT_FAMILY, COLOUR_BG, COLOUR_CARD_BG, COLOUR_CARD_BORDER,
                       COLOUR_TEXT, COLOUR_TEXT_MUTED, COLOUR_VERDICT)
import config


class LiveCameraController:
    """Free-run (continuous) camera acquisition - deliberately NOT
    software-triggered like camera_gui.py's CameraController. See
    module docstring for why this needs to be different."""

    def __init__(self, user_id, exposure=config.EXPOSURE_VAL):
        self.user_id = user_id
        self.cam = None
        self.exposure = exposure

    def connect(self, device_list):
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

        # Free-run / continuous acquisition - the key difference from
        # camera_gui.py, which sets TriggerMode=1 for software triggering.
        cam.MV_CC_SetEnumValue("TriggerMode", 0)  # 0 = trigger mode OFF (free-run)
        cam.MV_CC_SetFloatValue("ExposureTime", self.exposure)

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"[{self.user_id}] Start grabbing failed, ret [0x{ret:x}]")

        self.cam = cam

    def grab_frame(self, timeout_ms=500):
        """
        No software trigger command needed - the camera is already
        streaming continuously, this just pulls whatever the next
        available frame is. Returns None (not an error) if a frame
        isn't ready within the timeout - normal in free-run mode,
        just try again next poll.
        """
        if self.cam is None:
            raise RuntimeError(f"[{self.user_id}] Camera not connected")

        stFrame = MV_FRAME_OUT()
        memset(byref(stFrame), 0, sizeof(stFrame))
        ret = self.cam.MV_CC_GetImageBuffer(stFrame, timeout_ms)
        if ret != 0:
            return None

        width = stFrame.stFrameInfo.nWidth
        height = stFrame.stFrameInfo.nHeight
        buf_len = stFrame.stFrameInfo.nFrameLen
        buf = (c_ubyte * buf_len)()
        memmove(byref(buf), stFrame.pBufAddr, buf_len)
        frame = np.frombuffer(buf, dtype=np.uint8, count=width * height).reshape(
            (height, width)).copy()

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


class LiveDemoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Marula Shell Classifier - Live Demo")
        self.root.configure(bg=COLOUR_BG)
        self.root.minsize(900, 650)

        print("Loading model...")
        self.model, self.device, self.classes = load_model()
        print(f"Loaded model, classes = {self.classes}")

        self.cameras = {name: LiveCameraController(name) for name in config.CAMERA_NAMES}
        self.connected = {name: False for name in config.CAMERA_NAMES}
        self.running = False

        header = tk.Frame(root, bg=COLOUR_BG)
        header.pack(fill=tk.X, padx=24, pady=(20, 8))
        tk.Label(header, text="Marula Shell Classifier", bg=COLOUR_BG, fg=COLOUR_TEXT,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w")
        tk.Label(header, text="Live demo view", bg=COLOUR_BG, fg=COLOUR_TEXT_MUTED,
                 font=(FONT_FAMILY, 10)).pack(anchor="w")

        self.verdict_banner = tk.Frame(root, bg=COLOUR_VERDICT["WAITING"])
        self.verdict_banner.pack(fill=tk.X, padx=24, pady=16)
        self.verdict_var = tk.StringVar(value="WAITING")
        self.verdict_label = tk.Label(self.verdict_banner, textvariable=self.verdict_var,
                                      font=(FONT_FAMILY, 34, "bold"), fg="white",
                                      bg=COLOUR_VERDICT["WAITING"])
        self.verdict_label.pack(pady=(14, 0))
        self.verdict_detail_var = tk.StringVar(value="Click Start to begin streaming")
        self.verdict_detail_label = tk.Label(self.verdict_banner, textvariable=self.verdict_detail_var,
                                             font=(FONT_FAMILY, 10), fg="white",
                                             bg=COLOUR_VERDICT["WAITING"])
        self.verdict_detail_label.pack(pady=(2, 14))

        columns = tk.Frame(root, bg=COLOUR_BG)
        columns.pack(padx=24, pady=(0, 8))
        self.image_labels = {}
        self.camera_status_vars = {}
        for i, name in enumerate(config.CAMERA_NAMES):
            card = tk.Frame(columns, bg=COLOUR_CARD_BG, highlightbackground=COLOUR_CARD_BORDER,
                            highlightthickness=1, padx=14, pady=12)
            card.grid(row=0, column=i, padx=10)
            tk.Label(card, text=name, bg=COLOUR_CARD_BG, fg=COLOUR_TEXT,
                     font=(FONT_FAMILY, 13, "bold")).pack(anchor="w", pady=(0, 8))
            img_label = tk.Label(card, text="(not connected)", bg=COLOUR_CARD_BG, fg=COLOUR_TEXT_MUTED,
                                 font=(FONT_FAMILY, 10))
            img_label.pack()
            self.image_labels[name] = img_label
            status_var = tk.StringVar(value="Not connected")
            tk.Label(card, textvariable=status_var, bg=COLOUR_CARD_BG, fg=COLOUR_TEXT_MUTED,
                     font=(FONT_FAMILY, 9), wraplength=280, justify="left").pack(anchor="w", pady=(8, 0))
            self.camera_status_vars[name] = status_var

        self.fps_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.fps_var, bg=COLOUR_BG, fg=COLOUR_TEXT_MUTED,
                 font=("Consolas", 9)).pack(pady=(4, 8))

        btn_frame = tk.Frame(root, bg=COLOUR_BG)
        btn_frame.pack(pady=(4, 20))
        self.start_btn = tk.Button(btn_frame, text="Start Live View", command=self.on_start)
        self.start_btn.pack(side=tk.LEFT, padx=6)
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_start(self):
        try:
            device_list = MV_CC_DEVICE_INFO_LIST()
            ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
            if ret != 0:
                raise RuntimeError(f"Enum Devices failed, ret [0x{ret:x}]")
            if device_list.nDeviceNum == 0:
                raise RuntimeError("No GigE cameras found - check power/network connection")
        except Exception as e:
            print(f"ERROR: {e}")
            return

        failures = []
        for name, cam in self.cameras.items():
            try:
                cam.connect(device_list)
                self.connected[name] = True
                self.camera_status_vars[name].set("Connected - streaming")
            except Exception as e:
                self.connected[name] = False
                self.camera_status_vars[name].set(f"FAILED: {e}")
                failures.append(name)

        if failures:
            print(f"WARNING: failed to connect {failures} - live view will run with "
                  f"the remaining camera(s) only")

        if not any(self.connected.values()):
            print("ERROR: no cameras connected, cannot start live view")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._loop()

    def on_stop(self):
        self.running = False
        for cam in self.cameras.values():
            cam.disconnect()
        self.connected = {name: False for name in config.CAMERA_NAMES}
        for name in config.CAMERA_NAMES:
            self.camera_status_vars[name].set("Not connected")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._set_verdict("WAITING", "Stopped")

    def _loop(self):
        if not self.running:
            return

        loop_start = time.perf_counter()
        camera_results = {}
        for name, cam in self.cameras.items():
            if not self.connected[name]:
                continue
            frame = cam.grab_frame()
            if frame is None:
                continue  # no new frame ready this cycle - normal in free-run, just skip

            annotated, results, process_ms = process_frame(frame, self.model, self.device, self.classes)
            self._display_frame(name, annotated)
            camera_results[name] = results
            self.camera_status_vars[name].set(f"{len(results)} shell(s) detected  ({process_ms:.1f} ms)")

        self._update_verdict(camera_results)

        loop_ms = (time.perf_counter() - loop_start) * 1000
        fps = 1000 / loop_ms if loop_ms > 0 else 0
        self.fps_var.set(f"Loop: {loop_ms:.0f} ms  (~{fps:.1f} fps)")

        self.root.after(config.LIVE_POLL_INTERVAL_MS, self._loop)

    def _update_verdict(self, camera_results):
        # No shells anywhere = nothing placed yet - the normal resting
        # state through most of a live demo, not an error. Different
        # handling from camera_gui.py's trigger flow deliberately: there,
        # a 0-shell result from an actual button press IS meaningful (a
        # misfire worth flagging); here, it's just "waiting."
        if not camera_results or all(len(r) == 0 for r in camera_results.values()):
            self._set_verdict("WAITING", "Place a shell to begin")
            return

        # Still settling (e.g. a hand moving through frame, or only one
        # camera has picked it up yet) - don't flash ERROR at the
        # customer for every transient in-between frame, just wait.
        if any(len(r) == 0 for r in camera_results.values()):
            self._set_verdict("WAITING", "Detecting...")
            return

        verdict, explanation, _ = fuse_pass_fail(camera_results)
        self._set_verdict(verdict, explanation)

    def _set_verdict(self, verdict, explanation):
        colour = COLOUR_VERDICT.get(verdict, COLOUR_VERDICT["WAITING"])
        self.verdict_var.set(verdict)
        self.verdict_detail_var.set(explanation)
        self.verdict_banner.config(bg=colour)
        self.verdict_label.config(bg=colour)
        self.verdict_detail_label.config(bg=colour)

    def _display_frame(self, camera_name, annotated_bgr):
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((650, 500))
        tk_img = ImageTk.PhotoImage(pil_img)
        label = self.image_labels[camera_name]
        label.configure(image=tk_img, text="")
        label.image = tk_img  # keep a reference - Tkinter drops it otherwise

    def on_close(self):
        self.running = False
        for cam in self.cameras.values():
            cam.disconnect()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveDemoGUI(root)
    root.mainloop()