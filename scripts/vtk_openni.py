#!/usr/bin/python
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import vtk

class VtkPointCloud:

    def __init__(self, zMin=-10.0, zMax=10.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
        mapper.SetColorModeToDefault()
        mapper.SetScalarRange(zMin, zMax)
        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoint(self, point):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(point[2])
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()
        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')

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

def point_and_shoot(event, x, y, flags, param):
    global refPt, selecting
    if x > 640:
        x = x - 640
    if y > 480:
        y = y - 480
    if event == cv2.EVENT_LBUTTONDOWN:
        print "Mouse Down"
        refPt = [(x,y)]
        selecting = True
        print refPt
    elif event == cv2.EVENT_LBUTTONUP:
        print "Mouse Up"
        refPt.append((x,y))
        selecting = False
        print refPt
        # noting the other two vertices of the rectangle and printing
        refPt.append((refPt[1][0], refPt[0][1]))
        refPt.append((refPt[0][0], refPt[1][1]))
        print "The co-ordinates of ROI: ", refPt
        roi = depth_img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0], 1]
        print "Points of ROI: ", roi
        #print roi.shape 
        print "Mean of ROI: ", roi.mean()
        print "Max of ROI: ", roi.max()
        print "Min of ROI: ", roi.min()
        print "Standard Deviation of ROI: ", np.std(roi)
        print "Length of ROI: ", len(roi)


#def stats():
# Initial OpenCV Window Functions
cv2.namedWindow("Depth Image")
cv2.setMouseCallback("Depth Image", point_and_shoot)

pointCloud = VtkPointCloud()

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(pointCloud.vtkActor)
renderer.SetBackground(.2, .3, .4)
renderer.ResetCamera()

# Render Window
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)

# Interactor
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

# Loop
while True:
    pointCloud.clearPoints()
    # Grab a new depth frame
    depth_frame = depth_stream.read_frame()
    depth_frame_data = depth_frame.get_buffer_as_uint16()

    # Grab a new color frame
    color_frame = color_stream.read_frame()
    color_frame_data = color_frame.get_buffer_as_uint8()

    # Put the depth frame into a numpy array and reshape it
    depth_points = np.frombuffer(depth_frame_data, dtype=np.uint16)
    depth_img = depth_points
    depth_img.shape = (1, 480, 640)
    depth_img = np.concatenate((depth_img, depth_img, depth_img), axis=0)
    depth_img = np.swapaxes(depth_img, 0, 2)
    depth_img = np.swapaxes(depth_img, 0, 1)

    # Put the color frame into a numpy array, reshape it, and convert from bgr to rgb
    color_img = np.frombuffer(color_frame_data, dtype=np.uint8)
    color_img.shape = (480, 640, 3)
    color_img = color_img[...,::-1]

    if len(refPt) > 1:
        color_img= color_img.copy()
        cv2.rectangle(color_img, refPt[0], refPt[1], (0, 255, 0), 2)
        depth_img= depth_img.copy()
        cv2.rectangle(depth_img, refPt[0], refPt[1], (0, 255, 0), 2)
    depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))
    img = np.concatenate((color_img, depth_img), 1)


    for k in xrange(1000):
	point = 20*(np.random.rand(3)-0.5)
	pointCloud.addPoint(point)
#    for p in depth_points:
#        pointCloud.addPoint(p)


    # Begin Interaction
    renderWindow.Render()
        
    # Display the reshaped depth frame using OpenCV
    cv2.imshow("Depth Image", img)
    key = cv2.waitKey(1) & 0xFF

    # If the 'c' key is pressed, break the while loop
    if key == ord("c"):
        break

# Close all windows and unload the depth device
depth_frame.stop()
color_frame.stop()
openni2.unload()
cv2.destroyAllWindows()
