#!/usr/bin/python
import cv2
import numpy as np
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

# Function to return some pixel information when the OpenCV window is clicked
refPt = []
selecting = False
mousex = 0
mousey = 0


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


#def stats():
# Initial OpenCV Window Functions
cv2.namedWindow("Depth Image")
cv2.setMouseCallback("Depth Image", point_and_shoot)

# Loop
while True:
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

    # Display the reshaped depth frame using OpenCV
    cv2.imshow("Depth Image", img)

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        sys.exit(0)


# Close all windows and unload the depth device
depth_stream.stop()
color_stream.stop()
openni2.unload()
cv2.destroyAllWindows()
