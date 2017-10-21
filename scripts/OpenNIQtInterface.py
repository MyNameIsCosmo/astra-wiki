import sys
import cv2
import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
from OpenNIDevice import *

class CvToQImage(QtGui.QImage):
    def __init__(self, img, mapping="RGB"):
        height, width, channel = img.shape
        bytesPerLine = 3 * width
        self.mapping = mapping
        #self.qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        super(CvToQImage, self).__init__(img, width, height, QtGui.QImage.Format_RGB888)

class ImageView(QtGui.QLabel):
    def __init__(self, parent=None):
        super(QtGui.QLabel, self).__init__(parent)
        self._image = None

    def updateFrame(self, img, pixmap=False, height=640, width=480):
        self._image = img
        if self._image is not None:
            if pixmap:
                _image_pixmap = QtGui.QPixmap(self._image)
            else:
                _image_pixmap = QtGui.QPixmap.fromImage(self._image)
            _image_resized = _image_pixmap.scaled(height,width,Qt.KeepAspectRatio)
            self.setPixmap(_image_resized)

class ImageWidget(QtGui.QWidget):
    def __init__(self, parent, image=None, height=640, width=480):
        super(QtGui.QWidget, self).__init__(parent)
        layout = QVBoxLayout()
        self.imageLabel = ImageView(self)
        layout.addWidget(self.imageLabel)
        self.setLayout(layout)
        
        if image:
            self.update(image, height, width)

    def update(self, image, height=640, width=480):
	self.imageLabel.updateFrame(CvToQImage(cv_image),height=height,width=width)

class DeviceViewer(QtGui.QWidget):
    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.__widgets()
        self.__layout()

    def __widgets(self):
        #self.widget_image_color = ImageWidget(self)
        self.widget_image_color = QtGui.QLabel("Work in progress")

    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()

        self.hbox.addWidget(self.widget_image_color)

class DeviceSelection(QtGui.QWidget):
    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.__widgets()
        self.__layout()
        self._get_devices()
        self._list_devices()

    def __widgets(self):
        self.deviceListLabel = QtGui.QLabel("Select a Device")
        self.deviceList = QtGui.QListWidget()
        self.deviceList.clicked.connect(self._device_list_clicked)
        
    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        self.hbox = QtGui.QHBoxLayout()
        self.hbox2 = QtGui.QHBoxLayout()

        self.hbox.addWidget(self.deviceListLabel)
        self.hbox2.addWidget(self.deviceList)

        self.vbox.addLayout(self.hbox)
        self.vbox.addLayout(self.hbox2)
        self.setLayout(self.vbox)

    def _get_devices(self):
        self.devices = openni_list()

    def _list_devices(self):
        if len(self.devices) > 0:
            for d in self.devices:
                item = QtGui.QListWidgetItem("{} {}: {}".format(d[1], d[2], str(d[0])))
                self.deviceList.addItem(item)
        else:
            item = QtGui.QListWidgetItem("No Devices Detected!")
            self.deviceList.addItem(item)

    def _device_list_clicked(self, index):
        print self.devices[index.row()]


class MainWindow(QtGui.QMainWindow):

    def __init__(self, parent=None):
        super(QtGui.QMainWindow, self).__init__(parent)
        self._setWindowSize()
        self.__layout()
        self.__init_menu()
        self.statusBar().showMessage('Listing Devices')
        self.show_device_list()
        self.statusBar().showMessage('Ready')

    def __layout(self):
        _widget = QtGui.QWidget()
        self.mainWidget = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout(_widget)
        self.layout.addWidget(self.mainWidget)
        self.setCentralWidget(_widget)

    def __init_menu(self):
        self.mainMenu = self.menuBar()
        self.mainMenu.setNativeMenuBar(False)
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.dataMenu = self.mainMenu.addMenu('Data')
        self.aboutMenu = self.mainMenu.addMenu('About')
        self.debugMenu = self.mainMenu.addMenu('Debug')

        exitButton = QtGui.QAction(QtGui.QIcon(), 'Exit', self)
        exitButton.setShortcut("Ctrl+Q")
        exitButton.setStatusTip("Exit Application")
        exitButton.triggered.connect(self.close)
        self.fileMenu.addAction(exitButton)

    def _setWindowSize(self, width=800, height=600):
        self.resize(width, height)
        self.setMinimumSize(width, height)
        self.setMaximumSize(width, height)

    def _set_main_widget(self, cls):
        self.mainWidget.close()
        self.layout.removeWidget(self.mainWidget)
        self.mainWidget = cls(self)
        self.layout.addWidget(self.mainWidget)

    def show_device_list(self):
        self._set_main_widget(DeviceSelection)

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    main = MainWindow()
    main.show()
    #app.aboutToQuit.connect(main_window.destruct)
    app.exec_()
        #main_window.destruct()
