'''
TODO: 
    Camera intrinsics
    Camera position offset in 3d space
    Pop-up for grid properties
    Pop-up for camera properties
'''

import sys
from .QtCommon import *
from .Device import *
from .QtDeviceSelection import *
from .QtDeviceViewer import *
from .QtCV import *
from .QtPointCloud import *

from pyqtgraph.Qt import QtCore, QtGui

class DepthInterface(QtGui.QMainWindow):

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

    main = DepthInterface()
    main.show()
    app.aboutToQuit.connect(main._destruct)
    app.exec_()
