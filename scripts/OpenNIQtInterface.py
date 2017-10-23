import sys
import cv2
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from OpenNIDevice import *



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
        self._image = None

    def updateFrame(self, img, height=640, width=480, pixmap=False):
        self._image = img
        if self._image is not None:
            if pixmap:
                _image_pixmap = QtGui.QPixmap(self._image)
            else:
                _image_pixmap = QtGui.QPixmap.fromImage(self._image)
            _image_resized = _image_pixmap.scaled(height,width,QtCore.Qt.KeepAspectRatio)
            self.setPixmap(_image_resized)

class ImageWidget(QtGui.QWidget):

    def __init__(self, parent, image=None, height=640, width=480):
        super(QtGui.QWidget, self).__init__(parent)
        layout = QtGui.QVBoxLayout()
        self.imageLabel = ImageView(self)
        layout.addWidget(self.imageLabel)
        self.setLayout(layout)
        
        if image:
            self.update(image, height, width)

    def update(self, image, mapping=QtGui.QImage.Format_RGB888):
	self.imageLabel.updateFrame(CvToQImage(image, mapping=mapping))

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
        self.widget_image_color = ImageWidget(self)
        self.widget_image_color.setMaximumSize(640/2,480/2)
        self.widget_image_color.setMinimumSize(640/2,480/2)
        self.widget_image_depth = ImageWidget(self)
        self.widget_image_depth.setMaximumSize(640/2,480/2)
        self.widget_image_depth.setMinimumSize(640/2,480/2)

    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()

        self.hbox.addWidget(self.widget_image_color)
        self.hbox.addWidget(self.widget_image_depth)
        self.vbox.addLayout(self.hbox)

        self.setLayout(self.vbox)

    def _destroy(self):
        print "Destroying"
        if self.device:
            self.device.stop()
            print "Device stopped"
    
    def _update_images(self):
        image_color = self.device.get_frame_color()
        image_depth = self.device.get_frame_depth()

        image_depth = cv2.convertScaleAbs(image_depth, alpha=(255.0/65535.0))

        image_color = np.fliplr(image_color)
        image_depth = np.fliplr(image_depth)

        self.widget_image_color.update(image_color)
        self.widget_image_depth.update(image_depth, mapping=QtGui.QImage.Format_Indexed8)

    def _init_device(self, uri):
        self.device = OpenNIDevice(uri)
        self.device.open_stream_depth()
        self.device.open_stream_color()

        self.timer = QtCore.QTimer(self)
        self.timer.start(30) # FIXME: use 1000/FPS or whatever
        self.timer.timeout.connect(self._update_images)
        
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
        self._set_window_size()
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

    def _set_window_size(self, width=800, height=600):
        self.resize(width, height)
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)

    def _closeTab (self, currentIndex):
        #FIXME: Handle openni close
        currentQWidget = self.tabWidget.widget(currentIndex)
        currentQWidget._destroy()
        currentQWidget.deleteLater()
        self.tabWidget.removeTab(currentIndex)   

    def add_tab(self, cls, tooltip=None, closable=False, icon=None):
        try:
            name = cls.objectName()
        except Exception, e:
            print e
            name = "Unknown"
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
