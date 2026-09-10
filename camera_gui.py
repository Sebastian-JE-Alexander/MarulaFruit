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
from tkinter import messagebox, ttk
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

# EXPOSURE_VAL and CAMERA_NAMES live in config.py; visual design tokens
# (colours, fonts) live in gui_theme.py, shared with live_camera.py so
# both GUI scripts can't drift into different colours over time.
from gui_theme import (FONT_FAMILY, COLOUR_BG, COLOUR_CARD_BG, COLOUR_CARD_BORDER,
                       COLOUR_TEXT, COLOUR_TEXT_MUTED, COLOUR_ACCENT, COLOUR_VERDICT,
                       find_logo_path as _find_logo_path)


def find_logo_path():
    return _find_logo_path(config)


class CameraController:
    """
    One camera connect/trigger/grab/disconnect, identified by
    UserDefinedName - adapted from cameras.py's init_all_cameras() for
    software triggering instead of hardware.
    """

    def __init__(self, user_id, exposure=config.EXPOSURE_VAL):
        self.user_id = user_id
        self.cam = None
        self.exposure = exposure

    def connect(self, device_list):
        """device_list: an already-enumerated MV_CC_DEVICE_INFO_LIST,
        shared across all cameras being connected so enumeration only
        happens once per Connect click, not once per camera."""
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

        # --- Trigger config: SOFTWARE for this benchtop GUI ---
        # For the real belt integration, switch these two lines back to
        # cameras.py's hardware-trigger config instead:
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
        """Fires the software trigger, retrieves one frame, returns it
        as a (H,W) uint8 numpy array - matches what process_frame()
        expects. Assumes Mono8 (1 byte/pixel)."""
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
        self.root.configure(bg=COLOUR_BG)
        self.root.minsize(900, 650)

        self._setup_styles()

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

        # --- Header: logo + title -------------------------------------
        header = tk.Frame(root, bg=COLOUR_BG)
        header.pack(fill=tk.X, padx=24, pady=(20, 8))

        logo_path = find_logo_path()
        if logo_path:
            logo_img = Image.open(logo_path)
            logo_img.thumbnail((220, 80))
            self.logo_tk = ImageTk.PhotoImage(logo_img)  # kept as self. attr - Tkinter drops it otherwise
            tk.Label(header, image=self.logo_tk, bg=COLOUR_BG).pack(side=tk.LEFT, padx=(0, 16))
        else:
            print(f"No logo found in {config.LOGO_DIR}/ - skipping logo display "
                  f"(set config.LOGO_PATH explicitly, or add an image to that folder)")

        title_box = tk.Frame(header, bg=COLOUR_BG)
        title_box.pack(side=tk.LEFT, anchor="w")
        tk.Label(title_box, text="Marula Shell Classifier", bg=COLOUR_BG, fg=COLOUR_TEXT,
                 font=(FONT_FAMILY, 18, "bold")).pack(anchor="w")
        tk.Label(title_box, text="Live camera test", bg=COLOUR_BG, fg=COLOUR_TEXT_MUTED,
                 font=(FONT_FAMILY, 10)).pack(anchor="w")

        ttk.Separator(root, orient="horizontal").pack(fill=tk.X, padx=24, pady=(4, 0))

        # --- Verdict banner: the main demo output, a full-width coloured
        # block rather than just coloured text, so it reads clearly from
        # across a room during a live demo. Colour/text set in
        # show_verdict(); starts neutral grey before the first trigger.
        self.verdict_banner = tk.Frame(root, bg=COLOUR_VERDICT[None])
        self.verdict_banner.pack(fill=tk.X, padx=24, pady=16)
        self.verdict_var = tk.StringVar(value="—")
        self.verdict_label = tk.Label(self.verdict_banner, textvariable=self.verdict_var,
                                      font=(FONT_FAMILY, 34, "bold"),
                                      fg="white", bg=COLOUR_VERDICT[None])
        self.verdict_label.pack(pady=(14, 0))
        self.verdict_detail_var = tk.StringVar(value="Click Trigger to begin")
        self.verdict_detail_label = tk.Label(self.verdict_banner, textvariable=self.verdict_detail_var,
                                             font=(FONT_FAMILY, 10), fg="white", bg=COLOUR_VERDICT[None])
        self.verdict_detail_label.pack(pady=(2, 14))

        # --- One card per camera, side by side ---------------------------
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

            # NOTE: deliberately no width/height set here - Tkinter
            # measures Label width/height in CHARACTER units while
            # showing text but in PIXELS once only an image is
            # displayed, so a fixed value meant to size the text
            # placeholder collapses the box the moment an image
            # replaces the text. Auto-sizing avoids that entirely.
            img_label = tk.Label(card, text="(no image yet)", bg=COLOUR_CARD_BG, fg=COLOUR_TEXT_MUTED,
                                 font=(FONT_FAMILY, 10))
            img_label.pack()
            self.image_labels[name] = img_label

            status_var = tk.StringVar(value="Not connected")
            tk.Label(card, textvariable=status_var, bg=COLOUR_CARD_BG, fg=COLOUR_TEXT_MUTED,
                     font=(FONT_FAMILY, 9), wraplength=280, justify="left").pack(anchor="w", pady=(8, 0))
            self.camera_status_vars[name] = status_var

        # --- Timing (secondary info, deliberately understated) -----------
        self.timing_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.timing_var, bg=COLOUR_BG, fg=COLOUR_TEXT_MUTED,
                 font=("Consolas", 9)).pack(pady=(4, 8))

        # --- Buttons: Trigger is the primary action, larger and accented;
        # Connect/Save are secondary --------------------------------------
        btn_frame = tk.Frame(root, bg=COLOUR_BG)
        btn_frame.pack(pady=(4, 20))
        self.connect_btn = ttk.Button(btn_frame, text="Connect Cameras", style="Secondary.TButton",
                                      command=self.on_connect)
        self.connect_btn.pack(side=tk.LEFT, padx=6)
        self.trigger_btn = ttk.Button(btn_frame, text="Trigger", style="Primary.TButton",
                                      command=self.on_trigger, state=tk.DISABLED)
        self.trigger_btn.pack(side=tk.LEFT, padx=6)
        self.save_btn = ttk.Button(btn_frame, text="Save Capture", style="Secondary.TButton",
                                   command=self.on_save, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_styles(self):
        """ttk theming - 'clam' gives consistent cross-platform styling
        (unlike the default theme, which looks noticeably different per
        OS) that custom colours actually apply cleanly on top of."""
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Primary.TButton", font=(FONT_FAMILY, 11, "bold"),
                        foreground="white", background=COLOUR_ACCENT,
                        padding=(18, 10), borderwidth=0)
        style.map("Primary.TButton",
                  background=[("disabled", "#A9C2F5"), ("active", "#1D4ED8")])

        style.configure("Secondary.TButton", font=(FONT_FAMILY, 10),
                        foreground=COLOUR_TEXT, background="#E5E7EB",
                        padding=(14, 8), borderwidth=0)
        style.map("Secondary.TButton",
                  background=[("disabled", "#F3F4F6"), ("active", "#D1D5DB")])

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
                # you were just looking at, not keeping.
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
        colour = COLOUR_VERDICT.get(verdict, COLOUR_VERDICT[None])
        self.verdict_var.set(verdict)
        self.verdict_detail_var.set(explanation)
        self.verdict_banner.config(bg=colour)
        self.verdict_label.config(bg=colour)
        self.verdict_detail_label.config(bg=colour)

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