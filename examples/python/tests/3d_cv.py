#!/usr/bin/python
import cv2
import math
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

# An example using startStreams
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

#import pcl

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

dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

# Function to return some pixel information when the OpenCV window is clicked
refPt = []
distPt = None
selecting = False
calc_plane = True
done = False
mousex = 0
mousey = 0
drawing = False
plane_first = None
plane_second = None

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/(mdev if mdev else 1.)
    return data[s<m]

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def distance_to_plane(p, plane_data):
    plane, C = plane_data

    p0 = np.array(plane[0])
    p1 = np.array(plane[len(plane)/2])
    p2 = np.array(plane[-1])

    print p, p0, p1, p2

    # These two vectors are in the plane
    v1 = p2 - p0
    v2 = p1 - p0

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p2)

    u = p1 - p0
    v = p2 - p0
    # vector normal to plane
    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    p_ = p - p0
    dist_to_plane = np.dot(p_, n)
    p_normal = np.dot(p_, n) * n
    p_tangent = p_ - p_normal

    closest_point = p_tangent + p0
    coords = np.linalg.lstsq(np.column_stack((u, v)), p_tangent)[0]

    #x = closest_point[0]
    #y = closest_point[1]
    x = p[0]
    y = p[1]
    #linear z_plane = C[0]*x + C[1]*y + C[2] 
    z_plane = C[4]*x**2. + C[5]*y**2. + C[3]*x*y + C[1]*x + C[2]*y + C[0]
    
    pp = [x, y, z_plane]

    #dist_to_pp = math.sqrt((pp[0] - p[0]) ** 2 + (pp[1] - p[1]) ** 2 + (pp[2] - p[2]) ** 2)
    dist_to_pp = np.linalg.norm(pp-p)
    dist_to_pp = -dist_to_pp if pp[2] > p[2] else dist_to_pp

    print p
    print pp

    print "{}, {}".format(dist_to_plane, dist_to_pp)

    #return dist_to_plane
    return dist_to_pp

# Instrinsic calibration values guestimated.
#  Perform OpenCV monocular camera calibration for accurate depth data
def point_cloud(image_depth,image_color=None, cx=328, cy=241, fx=586, fy=589, scale=0.0001):
    pointcloud = np.zeros((1,1))
    colors = ((1.0, 1.0, 1.0, 1.0))
    depth = image_depth.astype(np.float32)
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0) & (depth < 65536.0)
    Z = np.where(valid, depth, np.nan)
    X = np.where(valid, (Z * (c - cx)) / fx, 0)
    Y = np.where(valid, (Z * (r - cy)) / fy, 0)
    pointcloud = np.dstack((X, Y, Z)) * scale

    if image_color is not None:
        colors = image_color.astype(np.float32)
        colors = np.divide(colors, 255) # values must be between 0.0 - 1.0
        colors = colors.reshape(colors.shape[0] * colors.shape[1], 3 ) # From: Rows X Cols X RGB -to- [[r,g,b],[r,g,b]...]
        colors = colors[...,::-1]

    return pointcloud, colors
         

def cv_mouse_event(event, x, y, flags, param):
    global refPt, done, selecting, mousex, mousey, drawing, distPt, calc_plane
    while drawing:
        continue
    if x > 640:
        x = x - 640
    if y > 480:
        y = y - 480
    mousex = x
    mousey = y
    """
    right-click = point
    left-click = draw shape

    """
    if event == cv2.EVENT_LBUTTONDOWN:
        #click
        if done and len(refPt) < 2:
            refPt = []
            distPt = None
            calc_plane = False
            done = False
            return
        if len(refPt) == 0:
            selecting = True
        if len(refPt) > 1 and done:
            #point
            distPt = (x,y)
        else:
            refPt.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        # click+drag
        if (refPt[0] != (x,y) and selecting):
            refPt.append((x,y))
            done = True
        selecting = False
    elif event == cv2.EVENT_RBUTTONDOWN:
        if not done:
            if len(refPt) > 2:
                done = True
            else:
                refPt = []
                distPt = None
                calc_plane = False
                done = False
                print "Need 3 or more points for polygon drawing!"
        else:
            refPt = []
            distPt = None
            calc_plane = False
            done = False

#QT app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 2
w.opts['azimuth'] = -90
w.opts['elevation'] = 0
w.opts['fov'] = 65
w.opts['center'] = QtGui.QVector3D(0.0, 1, 0.0)
w.show()
g = gl.GLGridItem()

w.addItem(g)

#initialize some points data
pos = np.zeros((10000,3))

sp2 = gl.GLScatterPlotItem(pos=pos)
sp2.scale(1,1,1)
sp2.rotate(-90,1,0,0)
sp2.setGLOptions('opaque')
w.addItem(sp2)

w2 = gl.GLViewWidget()
w2.opts['distance'] = 1
w2.opts['azimuth'] = -90
w2.opts['elevation'] = 0
w2.opts['fov'] = 65
w2.opts['center'] = QtGui.QVector3D(0.0, 1.0, 0.0)
w2.show()
g2 = gl.GLGridItem()

w2.addItem(g2)

sp3 = gl.GLScatterPlotItem(pos=np.zeros((10,3)))
sp3.scale(1,1,1)
sp3.rotate(-90,1,0,0)
sp3.setGLOptions('opaque')
w2.addItem(sp3)

sp4 = gl.GLSurfacePlotItem(x=np.empty((10,)), y=np.empty((10,)), z=np.empty((10, 10)), color=((0.0,0.0,1.0,0.4)))
sp4.scale(1,1,1)
sp4.rotate(-90,1,0,0)
sp4.setGLOptions('translucent')
w2.addItem(sp4)

sp5 = gl.GLSurfacePlotItem(x=np.empty((10,)), y=np.empty((10,)), z=np.empty((10, 10)), color=((1.0,0.0,0.0,0.4)))
sp5.scale(1,1,1)
sp5.rotate(-90,1,0,0)
sp5.setGLOptions('translucent')
w2.addItem(sp5)

#def stats():
# Initial OpenCV Window Functions
cv2.namedWindow("Depth Image", flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Depth Image", cv_mouse_event)

alpha = 0.5

# Loop
def update():
    global alpha, distPt, calc_plane, plane_first, plane_second

    if not calc_plane and not done:
        calc_plane = True
        plane_first = None
        plane_second = None

    '''
    Grab a frame, select the plane
    If the plane is selected, wait for key, then update frame and select the points
    If points are selected, get depth point from image
    '''

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

    depth_image = np.fliplr(depth_image)
    depth_img = np.fliplr(depth_img)
    color_img = np.fliplr(color_img)

    color_img = color_img.copy()
    depth_img = depth_img.copy()

    #shape_img= color_img.copy()
    #text_img= color_img.copy()
    #mask_img = color_img.copy()

    shape_img= np.zeros_like(color_img)
    text_img= np.zeros_like(color_img)
    mask_img= np.zeros_like(color_img)

    drawing = True

    try:
        # click only
        if len(refPt) == 1 and not selecting:
            point = (refPt[0][0],refPt[0][1])
            point_distance = float(depth_img[refPt[0][1]][refPt[0][0]][0]/10000.0)
            print point_distance
            cv2.circle(shape_img, refPt[0], 3, (0,0,255), 1)
            cv2.putText(text_img,"Point X,Y: {},{}".format(point[0], point[1]), (10,15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
            cv2.putText(text_img,"Point distance in Meters: {}".format(point_distance), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
        # click+drag
        if selecting and not done:
            cv2.rectangle(shape_img, refPt[0], (mousex,mousey), (0, 255, 0), 2)

        if len(refPt) > 1 and done:
            if len(refPt) == 2:
                mask = np.zeros((mask_img.shape[0], mask_img.shape[1]))
                cv2.rectangle(mask, refPt[0], refPt[1], 1, thickness=-1)
                mask = mask.astype(np.bool)

                mask_img[mask] = color_img[mask]

                cv2.rectangle(shape_img, refPt[0], refPt[1], (0, 255, 0), 2)
                left = min(refPt[0][0],refPt[1][0])
                top = min(refPt[0][1],refPt[1][1])
                right = max(refPt[0][0],refPt[1][0])
                bottom = max(refPt[0][1],refPt[1][1])

                #print "{}:{} {}:{}".format(top, left, bottom, right)
                roi = depth_image[top:bottom,left:right] # because the image is 480x640 not 640x480

        # polygon
        if len(refPt) > 1 and not done:
            cv2.polylines(shape_img, np.array([refPt]), False, (255,0,0), 3)
            cv2.line(shape_img, refPt[-1], (mousex, mousey), (0,255,0),3)

        if len(refPt) > 2 and done:
            cv2.polylines(shape_img, np.array([refPt]), True, (0,255,0), 3)

            mask = np.zeros((mask_img.shape[0], mask_img.shape[1]))
            points = np.array(refPt, dtype=np.int32)
            cv2.fillConvexPoly(mask, points, 1)
            mask = mask.astype(np.bool)

            mask_img[mask] = color_img[mask]

            left, top = tuple(points.min(axis=0))
            right, bottom = tuple(points.max(axis=0))

            cv2.rectangle(shape_img, (left, top), (right, bottom), color=(255,0,0), thickness=3)

            #print "{}:{} {}:{}".format(top, left, bottom, right)
            roi= depth_image[top:bottom,left:right] # because the image is 480x640 not 640x480


        # roi stats
        if len(refPt) > 1 and done:
            valid = (roi > 0) & (roi < 65536.0)
            roi = np.where(valid, roi, np.nan)

            roi_mean = np.round(np.nanmean(roi)/10000, 5)
            roi_max = np.round(np.nanmax(roi)/10000, 5)
            roi_min = np.round(np.nanmin(roi)/10000, 5)
            roi_std = np.round(np.nanstd(roi)/10000, 5)

            #cv2.rectangle(color_img, (5,5), (250,65), (50, 50, 50, 0), -1)
            cv2.putText(text_img,"Mean of ROI: {}".format(roi_mean), (10,15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
            cv2.putText(text_img,"Max of ROI: {}".format(roi_max), (10,30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
            cv2.putText(text_img,"Min of ROI: {}".format(roi_min), (10,45), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
            cv2.putText(text_img,"Standard Deviation of ROI: {}".format(roi_std), (10,60), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
            print "Mean of ROI: ", roi_mean
            print "Max of ROI: ", roi_max
            print "Min of ROI: ", roi_min
            print "Standard Deviation of ROI: ", roi_std

        if distPt is not None:
            cv2.circle(shape_img, distPt, 3, (0,0,255), 1)

        if plane_first is not None:
            C = np.round(plane_first[1],4)
            text_plane = '1 z= {}x^2 + {}y^2 + {}xy {}x + {}y + {}'.format(C[4], C[5], C[3], C[1], C[2], C[0])
            cv2.putText(text_img, text_plane, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))

        if plane_second is not None:
            C = np.round(plane_second[1],4)
            text_plane = '2 z= {}x^2 + {}y^2 + {}xy {}x + {}y + {}'.format(C[4], C[5], C[3], C[1], C[2], C[0])
            cv2.putText(text_img, text_plane, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))

    except Exception, e:
        print e


    drawing = False

    min_depth = 4500.0
    max_depth = float(max(np.nanmax(depth_img), 65535.0/4.0))
    alpha = float(min_depth/max_depth)
    depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/(65535.0/2.0)))

    cv2.addWeighted(mask_img, alpha, color_img, 1-alpha, 0, color_img)
    cv2.addWeighted(shape_img, 1, color_img, 1, 0, color_img)
    #cv2.addWeighted(mask_img, alpha, depth_img, 1-alpha, 0, depth_img)
    cv2.addWeighted(shape_img, alpha, depth_img, 1, 0, depth_img)

    depth_pt_image = depth_image.copy()
    color_pt_img = color_img.copy()
    #depth_pt_image = np.flipud(depth_image)
    #depth_pt_image = np.fliplr(depth_pt_image)
    #color_pt_img = np.flipud(color_img)
    #color_pt_img = np.fliplr(color_pt_img)

    cloud, colors = point_cloud(depth_pt_image, color_pt_img)

    # Calculate a dynamic vertex size based on window dimensions and camera's position - To become the "size" input for the scatterplot's setData() function.
    v_rate = 2.5 # Rate that vertex sizes will increase as zoom level increases (adjust this to any desired value).
    v_scale = np.float32(v_rate) / w.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
    v_offset = (w.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
    size = v_scale + v_offset

    x = cloud[:,:,0].flatten()
    y = cloud[:,:,1].flatten()
    z = cloud[:,:,2].flatten()

    N = max(x.shape)
    pos = np.empty((N,3))
    pos[:, 0] = x
    pos[:, 1] = y
    pos[:, 2] = z

    try:
        if len(refPt) > 1 and done:
            #roi_color = color_img[top:bottom,left:right] # because the image is 480x640 not 640x480

            roi_cloud = np.zeros_like(cloud)
            roi_colors = np.zeros_like(colors)

            cloud_mask = mask.flatten()
            color_mask = cloud_mask

            #roi_cloud[cloud_mask] = cloud[cloud_mask]
            roi_colors[color_mask] = colors[color_mask]

            roi_x = np.zeros_like(x) * np.nan
            roi_y = np.zeros_like(y) * np.nan
            roi_z = np.zeros_like(z) * np.nan

            roi_x[cloud_mask] = x[cloud_mask]
            roi_y[cloud_mask] = y[cloud_mask]
            roi_z[cloud_mask] = z[cloud_mask]

            N = max(x.shape)
            roi_points = np.empty((N,3))
            roi_points[:, 0] = roi_x
            roi_points[:, 1] = roi_y
            roi_points[:, 2] = roi_z 

            v2_rate = 2.5
            v2_scale = np.float32(v2_rate) / w2.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
            v2_offset = (w2.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
            size2 = v2_scale + v2_offset

            roi_data = np.c_[roi_x,roi_y,roi_z]
            finite = np.isfinite(roi_data).all(axis=1)
            roi_data = roi_data[finite]

            roi_colors = roi_colors[finite]

            sp3.setData(pos=roi_data, color=roi_colors, size=size2, pxMode=True) 

            plane = None

            # FIXME: outliers mess everything up
            #roi_data = reject_outliers(roi_data, 0.2)

            count = 50
            X = np.linspace(np.min(roi_data[:,0]), np.max(roi_data[:,0]), count)
            Y = np.linspace(np.min(roi_data[:,1]), np.max(roi_data[:,1]), count)
            Xx = X.reshape(count, 1)
            Yy = Y.reshape(1, count)

            calc = 2
            if calc == 1: #linear plane
                A = np.c_[roi_data[:,0], roi_data[:,1], np.ones(roi_data.shape[0])]
                C,_,_,_ = np.linalg.lstsq(A, roi_data[:,2])    # coefficients

                # evaluate it on grid
                Z = C[0]*Xx + C[1]*Yy + C[2]
                print "z = {}x + {}y + {}".format(np.round(C[0],4), np.round(C[1],4), np.round(C[2],4))

                z = C[0]*X + C[1]*Y + C[2]
                plane = (np.c_[X, Y, z], C)
            elif calc == 2: #quadratic plane
                A = np.c_[np.ones(roi_data.shape[0]), roi_data[:,:2], np.prod(roi_data[:,:2], axis=1), roi_data[:,:2]**2]
                C,_,_,_ = np.linalg.lstsq(A, roi_data[:,2])    # coefficients

                # evaluate it on grid
                Z = C[4]*Xx**2. + C[5]*Yy**2. + C[3]*Xx*Yy + C[1]*Xx + C[2]*Yy + C[0]
                print('z= {}x^2 + {}y^2 + {}xy {}x + {}y + {}'.format(C[4], C[5], C[3], C[1], C[2], C[0]))

                z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
                plane = (np.c_[X, Y, z], C)
            elif calc == 3: #lineplot
                # FIXME: This is not efficient, also want to update these line plots in real time to visualize deflection
                #for i in range(count):
                '''
                Loop through n segments with m resolution
                    take std of points from line to determine colors
                '''
                print "Meh"
                #y_resolution = 20
                #Y = np.linspace(np.min(roi_data[:,1]), np.max(roi_data[:,1]), y_resolution)
                #for i in range(y_resolution):
                    #x = roi_data[:,0]
                    #pts = np.vstack([x,yi,z]).transpose()
                    #sp5.setData(pos = pts, color=pg.glColor((i, n*1.3)), width=(i+1)/10.)

            
            if calc < 3:
                if calc_plane:
                    sp4.setData(x=X, y=Y, z=Z)
                else:
                    sp5.setData(x=X, y=Y, z=Z)

            if calc_plane:
                plane_first = plane
            else:
                plane_second = plane
            calc_plane = False

            if distPt is not None and not calc_plane:
            
                distance = "NaN"
                #if distPt[1] > top and distPt[1] < bottom:
                #    if distPt[0] < right and distPt[0] > left:
                # FIXME: find nearest cluster of points to selection
                pt = distPt
                p = cloud[pt[1]][pt[0]]
                #p = openni2.convert_depth_to_world(depth_stream, pt[0], pt[1], depth_image[pt[0]][pt[1]])
                #p = (p[0]/1000, p[1]/1000, p[2]/1000)
                distance = np.round(distance_to_plane(p, plane_first),4)
                cv2.putText(text_img,"Point Dist to Plane: {}".format(distance), (10,105), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255))
                #print "Distance to Plane: {}".format(distance)


    except Exception, e:
        print e

    cv2.addWeighted(text_img, 1, color_img, 1, 0, color_img)

    img = np.concatenate((color_img, depth_img), 1)

    # Display the reshaped depth frame using OpenCV
    cv2.imshow("Depth Image", img)

    sp2.setData(pos=pos, color=colors, size=size, pxMode=True)

    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
        depth_stream.stop()
        color_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()
        sys.exit(0)

t = QtCore.QTimer()
t.timeout.connect(update)
t.start(150)


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
