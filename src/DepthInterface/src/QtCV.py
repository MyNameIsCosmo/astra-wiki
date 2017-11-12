import cv2
import numpy as np

from .Common import *
from pyqtgraph.Qt import QtCore, QtGui

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


    def _update_frame(self, img, height=640, width=480, pixmap=False):
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
        self._update_frame(CvToQImage(image, mapping=mapping))

