#!/usr/bin/python
import sys
import cv2
import time
import logging
import ctypes
import io
import os
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from logging.config import dictConfig

# Class needed for redirecting ctype sysout to logger
#  https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
#  https://github.com/Supervisor/supervisor/blob/master/supervisor/loggers.py

logging_config = dict(
    version = 1,
    disable_existing_loggers = False, 
    formatters = {
        'simple': {'format':
              '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'}
        },
    handlers = {
        'console': {'class': 'logging.StreamHandler',
              'formatter': 'simple',
              'level': logging.DEBUG}
        },
    root = {
        'handlers': ['console'],
        'level': logging.DEBUG,
        },
)
dictConfig(logging_config)
logger = logging.getLogger('Device')
logger.debug("Rerouting stdout,stderr to logger")

# Initialize the depth device
openni2.initialize()
#openni2.configure_logging(severity=0, console=True)
dev = openni2.Device.open_any()

depth = False
color = True
ir = False

if depth:
    # Start the depth stream
    depth_stream = dev.create_depth_stream()
    depth_stream.start()
    depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

if color:
    # Start the color stream
    color_stream = dev.create_color_stream()
    color_stream.start()
    color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))

if ir:
    # Start the ir stream
    ir_stream = dev.create_ir_stream()
    ir_stream.start()
    #ir_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_GRAY16, resolutionX = 640, resolutionY = 480, fps = 30))
    video_modes = ir_stream.get_sensor_info().videoModes
    ir_stream.set_video_mode(video_modes[0])
    for mode in video_modes:
        print(mode)

# Loop
while True:
    if depth:
        # Grab a new depth frame
        depth_frame = depth_stream.read_frame()
        depth_frame_data = depth_frame.get_buffer_as_uint16()

        # Put the depth frame into a numpy array and reshape it
        depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
        depth_img.shape = (1, 480, 640)
        depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
        depth_img = np.swapaxes(depth_img, 0, 2)
        depth_img = np.swapaxes(depth_img, 0, 1)
        depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))
        # Display the reshaped depth frame using OpenCV
        cv2.imshow("Depth Image", depth_img)

    if  color:
        # Grab a new color frame
        color_frame = color_stream.read_frame()
        color_frame_data = color_frame.get_buffer_as_uint8()

        # Put the color frame into a numpy array, reshape it, and convert from bgr to rgb
        color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
        color_img.shape = (480, 640, 3)
        color_img = color_img[...,::-1]
        # Display the reshaped depth frame using OpenCV
        cv2.imshow("Color Image", color_img)

    if ir:
        ir_frame = ir_stream.read_frame()
        ir_frame_data = ir_frame.get_buffer_as_uint16()

        ir_img = np.frombuffer(ir_frame_data, dtype=np.uint16)
        ir_img.shape = (1, 480, 640)
        ir_img = np.concatenate((ir_img, ir_img, ir_img), axis=0)
        ir_img = np.swapaxes(ir_img, 0, 2)
        ir_img = np.swapaxes(ir_img, 0, 1)
        ir_img = cv2.convertScaleAbs(ir_img, alpha=(255.0/65535.0))
        cv2.imshow("IR Image", ir_img)

    key = cv2.waitKey(0) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        sys.exit(0)


# Close all windows and unload the depth device
if depth:
    depth_stream.stop()
if color:
    color_stream.stop()
if ir:
    ir_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
