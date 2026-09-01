"""
cameras.py

This script deals with how we interface with the Hikrobot GigE cameras that we will use for image acquisition.
This is a modified version of the code used and deployed for Yanfeng camera station
"""
# ----------------------------------------------------------------------------

import socket
import os
from pathlib import Path
import threading
import time
import queue

import numpy as np
import cv2
import tkinter as tk

from datetime import datetime
from tkinter import filedialog, scrolledtext, messagebox
from ctypes import *
from MvCameraControl_class import *
from MvErrorDefine_const import *
from CameraParams_header import *

# ============================ Configuration ========================================

PLC_IP = "192.168.10.181" # Adjust to match real plc
PLC_PORT = 502            # Adjust to match real plc
EXPOSURE_VAL = 40000.0    # float value

# =========================== Functions =============================================

def init_all_cameras():
    temp_list = [None] * 12
    device_list = MV_CC_DEVICE_INFO_LIST
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
    if ret != 0:
        print(f"Enum Devices fail! ret [0x{ret:x}]")
        return temp_list

    for i in range(device_list.nDeviceNum):
        try:
            st_device = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            user_id = "".join([chr(c) for c in st_device.SpecialInfo.stGigEInfo.chUserDefinedName if c != 0]).strip()

            if "CAM_" in user_id:
                idx = int(user_id.split('_')[1]) - 1
                if 0 <= idx < 12:
                    cam = MvCamera()
                    # 1. Create Handle
                    ret = cam.MV_CC_CreateHandle(st_device)
                    if ret != 0:
                        print(f"Handle create fail! {user_id} ret [0x{ret:x}]")
                        continue

                    # 2. Open Device
                    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
                    if ret != 0:
                        print(f"Open device fail! {user_id} ret [0x{ret:x}]")
                        continue

                    # 3. Configure Trigger settings
                    cam.MV_CC_SetEnumValue("TriggerMode", 1)        # sets the trigger to be hardware
                    cam.MV_CC_SetEnumValue("TriggerSource", 0)      # sets the trigger source to be line0
                    cam.MV_CC_SetEnumValue("TriggerActivation", 0)  # sets the trigger to be a rising edge
                    cam.MV_CC_SetFloatValue("ExposureTime", EXPOSURE_VAL)  # sets the exposure of the camera to a value that we determine


                    # 4. Start Grabbing
                    ret = cam.MV_CC_StartGrabbing()
                    if ret != 0:
                        print(f"Start grabbing fail! {user_id} ret [0x{ret:x}]")
                        continue

                    temp_list[idx] = cam
                    print(f"Mapped {user_id} successfully.")
        except Exception as e:
            print(f"Unexpected error on CAM index {i}: {e}")
    return temp_list

def close_all_cameras():
    if 'cameras' in locals():
        for cam in cameras:
            if cam:
                try:
                    cam.MV_CC_StopGrabbing()
                    cam.MV_CC_CloseDevice()
                    cam.MV_CC_DestroyHandle()
                except Exception as e:
                    print(f"Error during camera cleanup: {e}")

def open_tcp():
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.connect((PLC_IP, PLC_PORT))




