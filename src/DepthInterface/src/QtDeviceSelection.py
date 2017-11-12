from .Device import *
from .Common import *
from .QtDeviceViewer import *
 
class DeviceSelection(QtGui.QWidget):

    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.destroyed.connect(self.__destruct)
        self.setObjectName("Device Selection")
        self.parent_ = parent
        self.__widgets()
        self.__layout()
        self._list_devices()

        self.refreshTimer = QtCore.QTimer(self)
        self.refreshTimer.start(2000)
        self.refreshTimer.timeout.connect(self._list_devices)

    def __destruct(self):
        self.refreshTimer.stop()

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
                make = str(d[1], 'ascii')
                model = str(d[2], 'ascii')
                uri = str(d[0], 'ascii')
                item = QtGui.QListWidgetItem("{} {}: {}".format(make, model, uri))
                self.deviceList.addItem(item)
        else:
            item = QtGui.QListWidgetItem("No Devices Detected!")
            self.deviceList.addItem(item)

    def _device_list_clicked(self, index):
        device = self.devices[index.row()]
        tabToolTips = [self.parent_.tabWidget.tabToolTip(t) for t in range(self.parent_.tabWidget.count())]
        uri = device[0]
        if uri in tabToolTips:
            logger.debug("{} already open!".format(device[0]))
            self.parent_.statusBar().showMessage("{} already open in tab #{}".format(uri, tabToolTips.index(uri))) 
        else:
            device_view = DeviceViewer(self.parent_, device) 
            self.parent_.add_tab(device_view, device_view.uri, True)
            self.parent_.statusBar().showMessage("{} {} opened".format(device[1], device[2])) 

