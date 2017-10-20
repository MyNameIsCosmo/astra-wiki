#!/usr/bin/python
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

# An example using startStreams
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

# Initialize the depth device
openni2.initialize()
dev = openni2.Device.open_any()

'''
OpenNI Options:
    IMAGE_REGISTRATION_DEPTH_TO_COLOR
    IMAGE_REGISTRATION_OFF
Sensor Options:
    SENSOR_COLOR
    SENSOR_DEPTH
    SENSOR_IR
Pixel Format Options:
    PIXEL_FORMAT_DEPTH_100_UM
    PIXEL_FORMAT_DEPTH_1_MM
    PIXEL_FORMAT_GRAY16
    PIXEL_FORMAT_GRAY8
    PIXEL_FORMAT_JPEG
    PIXEL_FORMAT_RGB888
    PIXEL_FORMAT_SHIFT_9_2
    PIXEL_FORMAT_SHIFT_9_3
    PIXEL_FORMAT_YUV422
    PIXEL_FORMAT_YUYV
OpenNI Functions:
    configure_logging(directory=None, severity=None, console=None)
    convert_depth_to_color(depthStream, colorStream, depthX, depthY, depthZ)
    convert_depth_to_world(depthStream, depthX, depthY, depthZ)
    convert_world_to_depth(depthStream, worldX, worldY, worldZ)
    get_bytes_per_pixel(format)
    get_log_filename()
    get_version()
    initialize(dll_directories=['.'])
    is_initialized()
    unload()
    wait_for_any_stream(streams, timeout=None)
'''

class OpenNIDevice(openni2.Device):
    def __init__(self, uri=None, mode=None):
        openni2.Device.__init__(uri)
        self.stream_color = None
        self.stream_depth = None
        self.stream_ir = None
        self.stream_info = {
            'color': {'active': False, 'pointer': self.stream_color,
                  'x': 640, 'y': 480, 'fps': 30, 
                  'pixelFormat': PIXEL_FORMAT_RGB888},
            'depth': {'active': False, 'pointer': self.stream_depth,
                  'x': 640, 'y': 480, 'fps': 30, 
                  'pixelFormat': PIXEL_FORMAT_DEPTH_100_UM},
            'ir': {'active': False, 'pointer': self.stream_ir,
                   'x': 640, 'y': 480, 'fps': 30, 
                   'pixelFormat': c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM},
                       }  


    def open_depth_stream(self, x=640, y=480, fps=30, pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM):
        self.stream_info = {x, y, fps, pixelFormat}
        self.stream_depth = self.create_depth_stream()
        self.stream_depth.set_video_mode(openni2.VideoMode(pixelFormat = pixelFormat, 



# Start the depth stream
depth_stream = dev.create_depth_stream()
depth_stream.start()
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX = 640, resolutionY = 480, fps = 30))

# Start the color stream
color_stream = dev.create_color_stream()
color_stream.start()
color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX = 640, resolutionY = 480, fps = 30))

# Function to return some pixel information when the OpenCV window is clicked
refPt = []
selecting = False
mousex = 0
mousey = 0

# Instrinsic calibration values guestimated.
#  Perform OpenCV monocular camera calibration for accurate depth data
def point_cloud(depth, cx=328, cy=241, fx=586, fy=589, scale=0.0001):
    depth = depth.astype(np.float32)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 65536.0)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, (z * (c - cx)) / fx, 0)
    y = np.where(valid, (z * (r - cy)) / fy, 0)
    return x, y, z, np.dstack((x, y, z)) * scale

def point_and_shoot(event, x, y, flags, param):
    global refPt, selecting, mousex, mousey
    if x > 640:
        x = x - 640
    if y > 480:
        y = y - 480
    mousex = x
    mousey = y
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x,y)]
        selecting = True
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        if (refPt[0] != (x,y)):
            refPt.append((x,y))
        print refPt

#QT app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()

print w.cameraPosition()

w.addItem(g)

#initialize some points data
pos = np.zeros((10000,3))

sp2 = gl.GLScatterPlotItem(pos=pos)
sp2.scale(1,1,1)
w.addItem(sp2)

#def stats():
# Initial OpenCV Window Functions
cv2.namedWindow("Depth Image")
cv2.setMouseCallback("Depth Image", point_and_shoot)

# Loop
def update():
    
    colors = ((1.0, 1.0, 1.0, 1.0))
    # Grab a new depth frame
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()

    # Grab a new color frame
    color_frame = color_stream.read_frame()
    color_frame_data = color_frame.get_buffer_as_uint8()

    # Put the depth frame into a numpy array and reshape it
    depth_img = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img.shape = (1, 480, 640)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)

    depth_image = np.ndarray((depth_frame.height,depth_frame.width),dtype=np.uint16,buffer=depth_frame_data)

    # Put the color frame into a numpy array, reshape it, and convert from bgr to rgb
    color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
    color_img.shape = (480, 640, 3)
    color_img = color_img[...,::-1]

    color_img= color_img.copy()
    depth_img= depth_img.copy()
    try:
        if selecting:
            cv2.rectangle(color_img, refPt[0], (mousex,mousey), (0, 255, 0), 2)
            cv2.rectangle(depth_img, refPt[0], (mousex,mousey), (0, 255, 0), 2)
        if len(refPt) > 1:
            cv2.rectangle(color_img, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.rectangle(depth_img, refPt[0], refPt[1], (0, 255, 0), 2)
            left = min(refPt[0][0],refPt[1][0])
            top = min(refPt[0][1],refPt[1][1])
            right = max(refPt[0][0],refPt[1][0])
            bottom = max(refPt[0][1],refPt[1][1])
            roi = depth_image[top:left,bottom:right] # because the image is 480x640 not 640x480
            print "{}:{}, {}:{}".format(refPt[0][0], refPt[0][1], refPt[1][0], refPt[1][1])
            print "{}:{}, {}:{}".format(left, top, right, bottom)

            print roi.shape

            roi_mean = roi.mean()/10000
            roi_max = roi.max()/10000
            roi_min = roi.min()/10000
            roi_std = np.std(roi)/10000
            roi_std_from_mean = (np.std(roi) - roi_mean)


            cv2.putText(color_img,"Mean of ROI: {}".format(roi_mean), (10,15), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            cv2.putText(color_img,"Max of ROI: {}".format(roi_max), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            cv2.putText(color_img,"Min of ROI: {}".format(roi_min), (10,45), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            cv2.putText(color_img,"Standard Deviation of ROI: {}".format(roi_std_from_mean), (10,60), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            print "Mean of ROI: ", roi_mean
            print "Max of ROI: ", roi_max
            print "Min of ROI: ", roi_min
            print "Standard Deviation of ROI: ", roi_std_from_mean
        elif len(refPt) == 1:
            point = (refPt[0][0],refPt[0][1])
            point_distance = float(depth_img[refPt[0][1]][refPt[0][0]][0]/10000.0)
            print depth_image[refPt[0][1]][refPt[0][0]]
            cv2.circle(color_img, refPt[0], 3, (0,0,255), 1)
            cv2.circle(depth_img, refPt[0], 3, (0,0,255), 1)
            cv2.putText(color_img,"Point X,Y: {},{}".format(point[0], point[1]), (10,15), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
            cv2.putText(color_img,"Point distance in Meters: {}".format(point_distance), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
    except Exception, e:
        print e

    depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))

    img = np.concatenate((color_img, depth_img), 1)


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

    # Display the reshaped depth frame using OpenCV
    cv2.imshow("Depth Image", img)

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


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

# Close all windows and unload the depth device
depth_stream.stop()
color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
