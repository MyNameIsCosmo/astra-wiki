#!/usr/bin/python
import ctypes
import cv2
import numpy as np
from OpenNIDevice import *

if __name__ == "__main__":
    device = OpenNIDevice()

    
    cv2.namedWindow("Depth Image")
    cv2.namedWindow("Color Image")

    device.open_stream_depth()
    device.open_stream_color()

    while True:
        depth_img = device.get_frame_depth()
        depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))

        cv2.imshow("Depth Image", depth_img)

        color_img = device.get_frame_color()
        cv2.imshow("Color Image", color_img)


        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
            device.stop()
            break
    cv2.destroyAllWindows()

