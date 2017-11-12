from .Common import *
from .QtCV import *
from .QtPointCloud import *

class ImageDraw(ImageView):

    def __init__(self, parent=None):
        super(ImageView, self).__init__(parent)
        self.parent_ = parent
        self._image_mask = None
        self._image_shapes = None
        self._image_text = None

    def _handle_mouse_event(self, event, x, y, mouse_event):
        if event < 0:
            logger.info("{} released!".format(event*-1))
        else:
            logger.info("{} pressed!".format(event))

        logger.info("{}, {}".format(x, y))

    def mousePressEvent(self, QMouseEvent):
        self._handle_mouse_event(QMouseEvent.button(), QMouseEvent.x(), QMouseEvent.y(), QMouseEvent)

    def mouseReleaseEvent(self, QMouseEvent):
        self._handle_mouse_event(-1 * QMouseEvent.button(), QMouseEvent.x(), QMouseEvent.y(), QMouseEvent)

class DepthAnalyze(QtGui.QWidget):
    
    def __init__(self, parent=None, device=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.parent_ = parent
        self.setObjectName("Analyze")

        self.__language()
        self.__widgets()
        self.__layout()

    def __language(self):
        self.text_instruction = ("Left click to select point, Right click to verify/deselect.\n"
                                    "\tOne left click: View depth at point\n"
                                    "\tLeft click and drag: Create rectangle for region of interest\n"
                                    "\tLeft click and release: Draw a polygon for region of interest\n"
                                    "\tRight click: Set polgyon, or reset points")

    def __widgets(self):
        self.label_instructions = QtGui.QLabel(self.text_instruction)
        self.image_draw = ImageDraw(self)
        self.widget_image_color = ImageView(self)
        self.widget_image_depth = ImageView(self)
        self.widget_point_cloud = PointCloudViewer(self)

        self.widget_image_color.setObjectName("Color")
        self.widget_image_depth.setObjectName("Depth")
        self.widget_point_cloud.setObjectName("Points")
    
    def __layout(self):
        self.vbox = QtGui.QVBoxLayout()
        
        self.vbox.addWidget(self.label_instructions)

        self.setLayout(self.vbox)

    def update_images(self, image_depth = None, image_color=None):
        self.widget_image_color.update(image_color)
        self.widget_image_depth.update(image_depth)
        self.widget_point_cloud.update_plot(image_depth, image_color)
