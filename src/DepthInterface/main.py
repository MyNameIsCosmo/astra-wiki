'''
TODO: 
    Camera intrinsics
    Camera position offset in 3d space
    Pop-up for grid properties
    Pop-up for camera properties
'''
import sys
from .src import Common
from .src import QtDepthInterface

from pyqtgraph.Qt import QtCore, QtGui

def main(*args, **kwargs):
    app = QtGui.QApplication(sys.argv)

    main = QtDepthInterface.DepthInterface()
    app.aboutToQuit.connect(main._destruct)
    main.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
