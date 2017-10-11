# coding: utf-8

# An example using startStreams
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

import numpy as np
import cv2
import sys
from openni import openni2
from openni import _openni2 as c_api


# Initialize the depth device
openni2.initialize()
dev = openni2.Device.open_any()

# Start the depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX = 640, resolutionY = 480, fps = 30))

# Start the color stream
color_stream = dev.create_color_stream()
color_stream.start()
color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))

#undistorted = Frame(640, 480, 4)
#registered = Frame(640, 480, 4)

#QT app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

#initialize some points data
pos = np.zeros((10000,3))

sp2 = gl.GLScatterPlotItem(pos=pos)
sp2.scale(1,1,1)
w.addItem(sp2)

def point_cloud(depth, cx=640/2, cy=480/2, fx=60, fy=49.5, scale=0.001):
    depth = depth.astype(np.float32)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 65536.0)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, (z * (c - cx)) / fx, 0)
    y = np.where(valid, (z * (r - cy)) / fy, 0)
    return x, y, z, np.dstack((x, y, z)) * scale

def update():
    colors = ((1.0, 1.0, 1.0, 1.0))

    # Grab a new depth frame
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()

    # Grab a new color frame
    color_frame = color_stream.read_frame()
    color_frame_data = color_frame.get_buffer_as_uint8()

    depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img.shape = (1, 480, 640)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)

    #cloud = points.astype(np.float32)/1000
    depth_image = np.ndarray((depth_frame.height,depth_frame.width),dtype=np.uint16,buffer=depth_frame_data)
    x, y, z, cloud = point_cloud(depth_image)

    x = cloud[:,:,0].flatten()
    y = cloud[:,:,1].flatten()
    z = cloud[:,:,2].flatten()

    N = max(x.shape)
    pos = np.empty((N,3))
    pos[:, 0] = x
    pos[:, 1] = z
    pos[:, 2] = y

    #size = np.empty((pos.shape[0]))
    #color = np.empty((pos.shape[0], 4))

    #for i in range(pos.shape[0]):
    #    size[i] = 0.1
    #    color[i] = (1,0,0,1)

    #d = np.ndarray((depth_frame.height, depth_frame.width),dtype=np.uint16,buffer=points)#/10000
    out = pos

    color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
    color_img.shape = (480, 640, 3)
    color_img = color_img[...,::-1]

    cv_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))

    cv2.imshow("Image", color_img)
    sp2.setData(pos=out, size=2, color=colors, pxMode=True)

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        sys.exit(0)



t = QtCore.QTimer()
t.timeout.connect(update)
t.start(100)

cv2.namedWindow("Image")


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

depth_stream.stop()
color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()

sys.exit(0)
