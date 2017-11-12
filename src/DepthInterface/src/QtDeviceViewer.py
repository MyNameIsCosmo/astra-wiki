from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from .Device import *
from .Common import *
from .QtCV import *
from .QtPointCloud import *
from .QtDepthAnalyze import *

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
        self.tab_main = QtGui.QTabWidget()
        self.widget_images = QtGui.QWidget()
        #self.widget_info = QtGui.QWidget()
        self.widget_analyze = DepthAnalyze()
        #self.tab_images = QtGui.QTabWidget()
        #self.tab_point_cloud = QtGui.QTabWidget()

        self.tab_main.setTabPosition(0)
        #self.tab_images.setTabPosition(1)
        #self.tab_point_cloud.setTabPosition(1)

        self.widget_image_color = ImageView(self)
        self.widget_image_depth = ImageView(self)
        self.widget_point_cloud = PointCloudViewer(self)

        self.label_info = QtGui.QLabel("Work in progress")

        self.widget_image_color.setObjectName("Color")
        self.widget_image_depth.setObjectName("Depth")
        self.widget_point_cloud.setObjectName("Points")
        self.widget_images.setObjectName("Stream")
        self.label_info.setObjectName("Info")

        self.widget_point_cloud.setMinimumHeight(200)

    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        self.vbox_images = QtGui.QVBoxLayout(self.widget_images)

        self.vbox.setContentsMargins(0,0,0,0)
        self.vbox.setSpacing(5)

        self.vbox_images.addWidget(self.widget_image_color)
        self.vbox_images.addWidget(self.widget_image_depth)

        self._add_tab(self.tab_main, self.widget_images)
        self._add_tab(self.tab_main, self.widget_point_cloud)
        self._add_tab(self.tab_main, self.widget_analyze)
        self._add_tab(self.tab_main, self.label_info)

        self.vbox.addWidget(self.tab_main, 0, QtCore.Qt.AlignTop)
        #self.vbox.addWidget(self.tab_main, 0, QtCore.Qt.AlignTop)

        self.setLayout(self.vbox)

    def _destroy(self):
        if self.device:
            self.device.stop()
    
    def _update_images(self):
        # TODO: Have widgets pull 
        image_color = self.device.get_frame_color()
        image_depth = self.device.get_frame_depth()


        image_color = np.fliplr(image_color)
        image_depth = np.fliplr(image_depth)

        self.widget_point_cloud.update_plot(image_depth, image_color)

        # this needs to be done AFTER point cloud update
        image_depth = cv2.convertScaleAbs(image_depth, alpha=(255.0/65535.0))

        self.widget_image_color.update(image_color)
        self.widget_image_depth.update(image_depth, mapping=QtGui.QImage.Format_Indexed8)


        self.widget_analyze.update_images(image_depth, image_color)

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
        
