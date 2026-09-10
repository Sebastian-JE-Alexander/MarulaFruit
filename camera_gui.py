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

import csv
import glob
import os
import sys
import time
import tkinter as tk
from datetime import datetime
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

from detect_and_classify import load_model, process_frame, fuse_pass_fail
import config


# EXPOSURE_VAL and CAMERA_NAMES now live in config.py - update them
# there (not here) so this file and any future camera-related script
# stay in sync automatically.


def find_logo_path():
    """
    Returns config.LOGO_PATH if set and it exists, otherwise
    auto-detects the first image file in config.LOGO_DIR.
    Returns None if nothing is found, so the GUI can skip the
    logo rather than crashing on startup.
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


class CameraController:
    """
    One camera's connect/trigger/grab/disconnect, identified by
    UserDefinedName - adapted from cameras.py's init_all_cameras() for
    software triggering instead of hardware.
    """

    def __init__(self, user_id, exposure=config.EXPOSURE_VAL):
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

        # ------------------ Trigger config: SOFTWARE --------------------------
        # For the real integration, switch these two lines to
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

        self.cameras = {name: CameraController(name) for name in config.CAMERA_NAMES}
        self.connected = {name: False for name in config.CAMERA_NAMES}
        self.trigger_count = 0
        self.last_trigger_id = None
        self.last_trigger_data = {}  # camera_name -> capture data, populated by on_trigger, consumed by on_save

        os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
        os.makedirs(config.CAMERA_CAPTURES_DIR, exist_ok=True)

        logo_path = find_logo_path()
        if logo_path:
            logo_img = Image.open(logo_path)
            logo_img.thumbnail((300, 100))  # header-sized, not overwhelming the window
            self.logo_tk = ImageTk.PhotoImage(logo_img)  # kept as self. attr - Tkinter drops it otherwise
            tk.Label(root, image=self.logo_tk).pack(pady=(10, 0))
        else:
            print(f"No logo found in {config.LOGO_DIR}/ - skipping logo display "
                  f"(set config.LOGO_PATH explicitly, or add an image to that folder)")

        # PASS/FAIL readout - Font/colour set
        # in show_verdict() based on the result; starts neutral grey
        # with placeholder text before the first trigger.
        self.verdict_var = tk.StringVar(value="—")
        self.verdict_label = tk.Label(root, textvariable=self.verdict_var,
                                      font=("Arial", 36, "bold"), fg="gray")
        self.verdict_label.pack(pady=(10, 0))
        self.verdict_detail_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.verdict_detail_var, font=("Arial", 10)).pack()

        # One column per camera, side by side, each with its own image
        # and status line - so results can be compared at a glance.
        columns = tk.Frame(root)
        columns.pack(padx=10, pady=10)

        self.image_labels = {}
        self.camera_status_vars = {}
        for i, name in enumerate(config.CAMERA_NAMES):
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
        self.save_btn = tk.Button(btn_frame, text="Save Capture", command=self.on_save,
                                  state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

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
        # block the other.
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
        self.save_btn.config(state=tk.DISABLED)
        self.root.update()

        self.trigger_count += 1
        trigger_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.trigger_count:04d}"
        self.last_trigger_id = trigger_id
        self.last_trigger_data = {}

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

                # Stored, not saved to disk yet - on_save() writes this
                # out when (and only when) the Save Capture button is
                # pressed, so triggering to preview a result doesn't
                # silently fill up outputs/camera_captures/ with frames
                # you were just looking at.
                self.last_trigger_data[name] = {
                    "raw_frame": frame, "annotated": annotated, "results": results,
                    "grab_ms": grab_ms, "process_ms": process_ms,
                }

                n_good = sum(1 for r in results if r["class"] == "good")
                n_bad = sum(1 for r in results if r["class"] == "bad")
                self.camera_status_vars[name].set(
                    f"{len(results)} shell(s) - {n_good} good, {n_bad} bad "
                    f"(capture {grab_ms:.1f} ms, processing {process_ms:.1f} ms)")

            total_ms = (time.perf_counter() - total_start) * 1000
            per_cam_summary = "  |  ".join(f"{n}: {ms:.1f} ms" for n, ms in per_camera_ms.items())
            self.timing_var.set(f"{per_cam_summary}   ||   Total (both cameras): {total_ms:.1f} ms")

            camera_results = {name: data["results"] for name, data in self.last_trigger_data.items()}
            verdict, explanation, _ = fuse_pass_fail(camera_results)
            self.show_verdict(verdict, explanation)

            if self.last_trigger_data:
                self.save_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Trigger failed", str(e))
        finally:
            self.trigger_btn.config(state=tk.NORMAL)

    def show_verdict(self, verdict, explanation):
        colours = {"PASS": "green", "FAIL": "red", "ERROR": "orange"}
        self.verdict_var.set(verdict)
        self.verdict_label.config(fg=colours.get(verdict, "gray"))
        self.verdict_detail_var.set(explanation)

    def on_save(self):
        if not self.last_trigger_data:
            return
        saved_trigger_id = self.last_trigger_id
        for name, data in self.last_trigger_data.items():
            self.log_trigger_results(saved_trigger_id, name, data["raw_frame"],
                                     data["annotated"], data["results"],
                                     data["grab_ms"], data["process_ms"])
        messagebox.showinfo("Saved", f"Saved capture {saved_trigger_id} "
                                     f"({len(self.last_trigger_data)} camera(s)) to "
                                     f"{config.CAMERA_CAPTURES_DIR}/ and {config.CAMERA_RESULTS_LOG_PATH}")
        # Clear the stored data itself, not just the button - the button
        # state alone only stops real clicks, not a stray second call to
        # this method by any other path. Clearing the data makes the
        # guard at the top of this function actually effective either way.
        self.last_trigger_data = {}
        self.save_btn.config(state=tk.DISABLED)

    def log_trigger_results(self, trigger_id, camera_name, raw_frame, annotated_frame,
                            results, grab_ms, process_ms):
        """
        Saves the raw + annotated frame for this camera/trigger to
        outputs/camera_captures/, and appends one CSV row per detected
        shell to outputs/camera_results_log.csv (one row with class=""
        if zero shells were found, so a trigger with no detections still
        shows up in the log rather than silently vanishing).
        """
        raw_path = os.path.join(config.CAMERA_CAPTURES_DIR,
                                f"{trigger_id}_{camera_name}_raw.png")
        annotated_path = os.path.join(config.CAMERA_CAPTURES_DIR,
                                      f"{trigger_id}_{camera_name}_annotated.png")
        cv2.imwrite(raw_path, raw_frame)
        cv2.imwrite(annotated_path, annotated_frame)

        file_exists = os.path.isfile(config.CAMERA_RESULTS_LOG_PATH)
        fieldnames = ["timestamp", "trigger_id", "camera", "shell_index", "class",
                      "confidence", "bbox", "capture_ms", "processing_ms",
                      "raw_path", "annotated_path"]
        with open(config.CAMERA_RESULTS_LOG_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            rows = results if results else [None]  # log a placeholder row for zero-shell triggers
            for i, r in enumerate(rows):
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "trigger_id": trigger_id,
                    "camera": camera_name,
                    "shell_index": i if r else "",
                    "class": r["class"] if r else "",
                    "confidence": round(r["confidence"], 4) if r else "",
                    "bbox": r["bbox"] if r else "",
                    "capture_ms": round(grab_ms, 2),
                    "processing_ms": round(process_ms, 2),
                    "raw_path": raw_path,
                    "annotated_path": annotated_path,
                })

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