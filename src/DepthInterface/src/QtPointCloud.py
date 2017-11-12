from .Common import *
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


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

    def _depth_image_to_point_cloud(self, image_depth, image_color=None, cx=328, cy=241, fx=586, fy=589, scale=0.0001):
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
            v_rate = 2.5 # Rate that vertex sizes will increase as zoom level increases (adjust this to any desired value).
            v_scale = np.float32(v_rate) / self.opts['distance'] # Vertex size increases as the camera is "zoomed" towards center of view.
            v_offset = (self.geometry().width() / 1000)**2 # Vertex size is offset based on actual width of the viewport.
            size = v_scale + v_offset

        x = cloud[:,:,0].flatten()
        y = cloud[:,:,1].flatten()
        z = cloud[:,:,2].flatten()
        
        N = max(x.shape)
        pos = np.empty((N,3))
        pos[:,0] = x
        pos[:,1] = y
        pos[:,2] = z

        self.scatterPlot.setData(pos=pos, color=colors, size=size, pxMode=pxMode)
    
    def depth_to_cloud(self, image_depth, image_color=None):
        cloud = self._depth_image_to_point_cloud(image_depth, image_color)
        return cloud

