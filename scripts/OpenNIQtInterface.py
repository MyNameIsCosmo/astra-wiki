'''
TODO: 
    Camera intrinsics
    Camera position offset in 3d space
    Pop-up for grid properties
    Pop-up for camera properties
'''

import sys
import cv2
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from OpenNIDevice import *

#import pcl

import time

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class PointCloudViewer(gl.GLViewWidget):

    def __init__(self, parent=None):
        gl.GLViewWidget.__init__(self, parent)
        #super(gl.GLViewWidget, self).__init__(parent)
        self.parent_ = parent
        self.__opts()
        self.__widgets()
        self.__layout()

    def __opts(self):
        self.opts['distance'] = 2
        self.opts['azimuth'] = -90
        self.opts['elevation'] = 0
        self.opts['fov'] = 65
        self.opts['center'] = QtGui.QVector3D(0.0, 0.75, 0.0)

    def __widgets(self):
        self.grid = gl.GLGridItem()
        self.scatterPlot = gl.GLScatterPlotItem(pos=np.zeros((10000,3)))
        self.scatterPlot.rotate(-90,1,0,0)
        self.scatterPlot.scale(1,1,1)
        self.scatterPlot.setGLOptions('opaque')

    def __layout(self):
        self.addItem(self.grid)
        self.addItem(self.scatterPlot)

    def _depth_image_to_point_cloud(self, image_depth, image_color=None, cx=328, cy=241, fx=586, fy=589, scale=0.1):
        #FIXME: scale appropriately to depth_image size, 1mm or 100um
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
            #colors = colors[:, :3:]  # remove alpha (fourth index) from BGRA to BGR
            #colors = colors[...,::-1] #BGR to RGB

        return pointcloud, colors
    
    def update_plot(self, image_depth=None, image_color=None, cloud=None, size=None, pxMode=True):
        if cloud is None:
            if image_depth is None:
                raise TypeError('A depth image or point cloud are needed to update the scatter plot!')
            cloud, colors = self._depth_image_to_point_cloud(image_depth, image_color)
        else:
            colors = ((1.0, 1.0, 1.0, 1.0))

        if size is None:
            # Calculate a dynamic vertex size based on window dimensions and camera's position - To become the "size" input for the scatterplot's setData() function.
            v_rate = 5.0 # Rate that vertex sizes will increase as zoom level increases (adjust this to any desired value).
            v_scale = np.float32(v_rate) / self.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
            v_offset = (self.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
            size = v_scale + v_offset

        try:
            with Timer() as t:
                x = cloud[:,:,0].flatten()
                y = cloud[:,:,1].flatten()
                z = cloud[:,:,2].flatten()
                
                N = max(x.shape)
                pos = np.empty((N,3))
                pos[:,0] = x
                pos[:,1] = y
                pos[:,2] = z

                self.scatterPlot.setData(pos=pos, color=colors, size=size, pxMode=pxMode)
        finally:
            pass
            #TODO: print debug
            #print('Point Cloud Rendering took %.03f sec.' % t.interval)
    
    def depth_to_cloud(self, image_depth, image_color=None):
        cloud = self._depth_image_to_point_cloud(image_depth, image_color)
        return cloud

class CvToQImage(QtGui.QImage):

    def __init__(self, img, mapping=QtGui.QImage.Format_RGB888):
        #FIXME: map "mapping" values to qimage formats
        if len(img.shape) > 2:
            height, width, channel = img.shape
        else:
            height, width = img.shape
        bytesPerLine = 3 * width
        #self.qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        super(CvToQImage, self).__init__(img.tostring(), width, height, mapping)

class ImageView(QtGui.QLabel):
    def __init__(self, parent=None):
        super(QtGui.QLabel, self).__init__(parent)
        self.parent_ = parent
        self._image = None

    def mousePressEvent(self, QMouseEvent):
        print QMouseEvent.pos()

    def mouseReleaseEvent(self, QMouseEvent):
        print QMouseEvent.pos()

    def _updateFrame(self, img, height=640, width=480, pixmap=False):
        self._image = img
        if self._image is not None:
            if pixmap:
                _image_pixmap = QtGui.QPixmap(self._image)
            else:
                _image_pixmap = QtGui.QPixmap.fromImage(self._image)
            if self.parent_ is not None:
                height = self.parent_.geometry().height()
                width = self.parent_.geometry().width()
                _image_pixmap = _image_pixmap.scaled(height,width,QtCore.Qt.KeepAspectRatio)
            self.setPixmap(_image_pixmap)

    def update(self, image, mapping=QtGui.QImage.Format_RGB888):
	self._updateFrame(CvToQImage(image, mapping=mapping))

class DeviceViewer(QtGui.QWidget):

    def __init__(self, parent=None, device=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.destroyed.connect(self._destroy)
        self.parent_ = parent
        self.uri, self.make, self.model = device
        self.setObjectName("{} {}".format(self.make, self.model))

        self.__widgets()
        self.__layout()
        self._init_device(self.uri)

    def __widgets(self):
        # TODO: make a QTabWidget object to handle this
        self.tab_images = QtGui.QTabWidget()
        self.tab_point_cloud = QtGui.QTabWidget()

        self.tab_images.setTabPosition(1)
        self.tab_point_cloud.setTabPosition(1)

        self.widget_image_color = ImageView(self)
        self.widget_image_depth = ImageView(self)
        self.widget_point_cloud = PointCloudViewer(self)

        self.widget_image_color.setObjectName("Color")
        self.widget_image_depth.setObjectName("Depth")
        self.widget_point_cloud.setObjectName("Points")

        self.widget_point_cloud.setMinimumHeight(200)

    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()

        self.vbox.setContentsMargins(0,0,0,0)
        self.vbox.setSpacing(0)

        self._add_tab(self.tab_point_cloud, self.widget_point_cloud)
        self._add_tab(self.tab_images, self.widget_image_color)
        self._add_tab(self.tab_images, self.widget_image_depth)


        self.vbox.addWidget(self.tab_point_cloud, 0, QtCore.Qt.AlignTop)
        self.vbox.addWidget(self.tab_images, 0, QtCore.Qt.AlignTop)

        self.setLayout(self.vbox)

    def _destroy(self):
        if self.device:
            self.device.stop()
    
    def _update_images(self):
        image_color = self.device.get_frame_color()
        image_depth = self.device.get_frame_depth()

        image_depth = cv2.convertScaleAbs(image_depth, alpha=(255.0/65535.0))

        image_color = np.fliplr(image_color)
        image_depth = np.fliplr(image_depth)

        self.widget_image_color.update(image_color)
        self.widget_image_depth.update(image_depth, mapping=QtGui.QImage.Format_Indexed8)

        self.widget_point_cloud.update_plot(image_depth, image_color)

    def _init_device(self, uri):
        self.device = OpenNIDevice(uri)
        self.device.open_stream_depth()
        self.device.open_stream_color()

        self.device.depth_color_sync = True
        self.device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        self.timer = QtCore.QTimer(self)
        self.timer.start(30) # FIXME: use 1000/FPS or whatever
        self.timer.timeout.connect(self._update_images)

    def _add_tab(self, tab_widget, cls, tooltip=None, closable=False, icon=None):
        name = "Unknown"
        if hasattr(cls, "objectName"):
            name = cls.objectName()
        tab = tab_widget.addTab(cls, name)
        if tooltip:
            tab_widget.setTabToolTip(tab, tooltip)
        if not closable:
            tab_widget.tabBar().setTabButton(tab, QtGui.QTabBar.RightSide,None)
        if icon:
            tab_widget.setTabIcon(icon)
        
class DeviceSelection(QtGui.QWidget):

    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.setObjectName("Device Selection")
        self.parent_ = parent
        self.__widgets()
        self.__layout()
        self._list_devices()

    def __widgets(self):
        self.deviceListLabel = QtGui.QLabel("Select a Device")
        self.deviceList = QtGui.QListWidget()
        self.deviceList.clicked.connect(self._device_list_clicked)
        
    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()
        self.hbox2 = QtGui.QHBoxLayout()

        self.button_refresh = QtGui.QPushButton("Refresh")
        self.button_refresh.clicked.connect(self._list_devices)
        self.button_refresh.setFixedWidth(120)

        self.hbox.addWidget(self.deviceListLabel)
        self.hbox.addItem(QtGui.QSpacerItem(1,1,QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum))
        self.hbox.addWidget(self.button_refresh)
        self.hbox2.addWidget(self.deviceList)

        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox2)
        self.setLayout(self.vbox)

    def _list_devices(self):
        self.deviceList.clear()
        self.devices = openni_list()
        if len(self.devices) > 0:
            for d in self.devices:
                item = QtGui.QListWidgetItem("{} {}: {}".format(d[1], d[2], str(d[0])))
                self.deviceList.addItem(item)
        else:
            item = QtGui.QListWidgetItem("No Devices Detected!")
            self.deviceList.addItem(item)

    def _device_list_clicked(self, index):
        device = self.devices[index.row()]
        tabToolTips = [unicode(self.parent_.tabWidget.tabToolTip(t)) for t in range(self.parent_.tabWidget.count())]
        uri = unicode(device[0])
        if uri in tabToolTips:
            print "{} already open!".format(device[0])
            self.parent_.statusBar().showMessage("{} already open in tab #{}".format(uri, tabToolTips.index(uri))) 
        else:
            device_view = DeviceViewer(self.parent_, device) 
            self.parent_.add_tab(device_view, device_view.uri, True)
            self.parent_.statusBar().showMessage("{} {} opened".format(device[1], device[2])) 


class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(QtGui.QMainWindow, self).__init__(parent)
        self._set_window_size(800,600,False)
        self.__layout()
        self.__init_menu()
        self.statusBar().showMessage('Listing Devices')
        self.show_device_list()
        self.statusBar().showMessage('Ready')

    def __layout(self):
        _widget = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(_widget)

        self.tabWidget = QtGui.QTabWidget()
        self.tabWidget.setTabPosition(1)
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.tabCloseRequested.connect(self._closeTab)

        self.layout.addWidget(self.tabWidget)
        self.setCentralWidget(_widget)

    def __init_menu(self):
        self.mainMenu = self.menuBar()
        self.mainMenu.setNativeMenuBar(False)
        self.fileMenu = self.mainMenu.addMenu('&File')
        #self.dataMenu = self.mainMenu.addMenu('Data')
        self.aboutMenu = self.mainMenu.addMenu('About')
        #self.debugMenu = self.mainMenu.addMenu('Debug')

        exitButton = QtGui.QAction(QtGui.QIcon(), 'Exit', self)
        exitButton.setShortcut("Ctrl+Q")
        exitButton.setStatusTip("Exit Application")
        exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(exitButton)

    def _destruct(self):
        print "Destruction TODO"

    def _set_window_size(self, width=800, height=600, resizable=False):
        self.resize(width, height)
        if not resizable:
            self.setMinimumSize(width, height)
            self.setMaximumSize(width, height)

    def _closeTab (self, currentIndex):
        currentQWidget = self.tabWidget.widget(currentIndex)
        currentQWidget._destroy()
        currentQWidget.deleteLater()
        self.tabWidget.removeTab(currentIndex)   

    def add_tab(self, cls, tooltip=None, closable=False, icon=None):
        name = "Unknown"
        if hasattr(cls, "objectName"):
            name = cls.objectName()
        tab = self.tabWidget.addTab(cls, name)
        if tooltip:
            self.tabWidget.setTabToolTip(tab, tooltip)
        if not closable:
            self.tabWidget.tabBar().setTabButton(tab, QtGui.QTabBar.RightSide,None)
        if icon:
            self.tabWidget.setTabIcon(icon)
	self.tabWidget.setCurrentIndex(tab)

    def show_device_list(self):
        device_list = DeviceSelection(self)
        self.add_tab(device_list)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = MainWindow()
    main.show()
    app.aboutToQuit.connect(main._destruct)
    app.exec_()
