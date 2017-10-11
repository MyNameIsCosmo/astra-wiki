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
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

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
sp2.scale(.1,.1,.1)
w.addItem(sp2)

def update():
    colors = ((1.0, 1.0, 1.0, 1.0))

    # Grab a new depth frame
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()

    # Grab a new color frame
    color_frame = color_stream.read_frame()
    color_frame_data = color_frame.get_buffer_as_uint8()

    points = np.frombuffer(depth_frame_data, dtype=np.uint16)
    points.shape = (1, 480, 640)
    points = np.concatenate((points,points,points), axis=0)
    points = np.swapaxes(points, 0, 2)
    points = np.swapaxes(points, 0, 1)
    # (x,y,z)

    d = np.ndarray((depth_frame.height, depth_frame.width),dtype=np.uint16,buffer=depth_frame_data)#/10000
    for row in range(depth_frame.height)
        for col in range(depth_frame.width)
            X = row
            Y = col
            Z = d[row][col]

    #out = pos

    cv_img = cv2.convertScaleAbs(points, alpha=(255.0/65535.0))

    cv2.imshow("Image", cv_img)
    sp2.setData(pos=out, size=1, color=colors, pxMode=True)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
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
