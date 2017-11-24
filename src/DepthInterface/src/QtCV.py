# https://stackoverflow.com/questions/26791357/pyqt-qgraphicsview-pan-and-zoom-with-transform-matrix
# https://stackoverflow.com/questions/3779654/pyqt-function-for-zooming-large-images#
from .Common import *

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

class ImageLabel(QtGui.QLabel):
    def __init__(self, parent=None):
        super(QtGui.QLabel, self).__init__(parent)
        self.parent_ = parent
        self._image = None
        self.resize(640, 480)

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

class ImageScene(QtGui.QGraphicsScene):
    def __init__(self, parent, view):
        super(QtGui.QGraphicsScene, self).__init__(parent)
        self.parent_ = parent
        self.view = view

        self.label = ImageLabel()
        self.addWidget(self.label)

        self.start_pos = QtCore.QPoint(0,0)

    def mousePressEvent(self, event):
        self.view.setDragMode(1) # Works fine without this
        self.start_pos=event.scenePos()

    # FIXME: Unusued??
    def mouseReleaseEvent(self, event):
        self.view.setDragMode(0)

    def wheelEvent(self, event):
        sc=event.delta()/100
        if sc<0: sc=-1/sc
        self.view.scale(sc,sc)
        self.view.setDragMode(0)

    def update(self, image, mapping=QtGui.QImage.Format_RGB888):
        self.label.update(image, mapping)
          
class ImageView(QtGui.QGraphicsView):
    
    def __init__(self, parent=None):
        super(QtGui.QGraphicsView, self).__init__(parent)
        self.parent_ = parent

        self.scene = ImageScene(self, self)
        self.setScene(self.scene)

    def mouseMoveEvent(self,event):
        if event.buttons() == QtCore.Qt.LeftButton:
            pos=event.pos()
            pos=self.mapToScene(pos)
            dx=pos.x()-self.scene.start_pos.x()
            dy=pos.y()-self.scene.start_pos.y()

            rect=self.sceneRect().getRect()
            self.setSceneRect(rect[0]-dx,rect[1]-dy,rect[2],rect[3])

    def resetM(self):
        self.resetTransform()

    def update(self, image, mapping=QtGui.QImage.Format_RGB888):
        self.scene.update(image, mapping)


