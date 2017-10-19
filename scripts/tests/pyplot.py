#!/usr/bin/env python
import os
import sys
import ctypes
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import matplotlib.pyplot as plt

#Initialized OpenNI
openni2.initialize() 
status = openni2.is_initialized()
if status:
    print 'OpenNI2 Initialization Done !'
else:
    print 'OpenNI2 Initialization Failed !'

#Open Device
dev = openni2.Device.open_any()


depth_stream = dev.create_depth_stream()

print 'device info:',dev.get_device_info()
print 'rgb video mode:', depth_stream.get_video_mode() 


#depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
depth_stream.start()


while(1):
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()
    depth_array = np.ndarray((depth_frame.height,depth_frame.width),dtype=np.uint8,buffer=depth_frame_data) 
    depth_array = np.ndarray((depth_frame.height,depth_frame.width),dtype=np.uint16,buffer=depth_frame_data) 


    #img = np.fromstring(depth_frame.get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
    plt.imshow(depth_array)
    plt.show()
#    cv2.imshow(img)
#    cv2.waitKey()

    

