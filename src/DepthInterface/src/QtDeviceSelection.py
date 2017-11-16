from .Common import *
from .Device import *
from .QtDeviceViewer import *
 
class DeviceSelection(QtGui.QWidget):

    def __init__(self, parent=None):
        super(QtGui.QWidget, self).__init__(parent)
        self.parent_ = parent
        self.destroyed.connect(self.__destruct)
        self.setObjectName("Device Selection")

        self.openni_path = self.parent_.args['openni']

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
        self.devices = openni_list(self.openni_path)
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

    def _open_device(self, device):
        device_view = DeviceViewer(self.parent_, device) 
        self.parent_.add_tab(device_view, device_view.uri, True)
        make = str(device[1], 'ascii')
        model = str(device[2], 'ascii')
        debugMsg = "{} {} opened".format(make, model)
        self.parent_.statusBar().showMessage(debugMsg) 
        logging.debug(debugMsg)

    def _device_open(self, device):
        tabToolTips = [self.parent_.tabWidget.tabToolTip(t) for t in range(self.parent_.tabWidget.count())]
        uri = str(device[0], 'ascii')
        if uri in tabToolTips:
            logger.debug("{} already open!".format(uri))
            self.parent_.statusBar().showMessage("{} already open in tab #{}".format(uri, tabToolTips.index(uri))) 
            return True
        return False

    def _device_list_clicked(self, index):
        device = self.devices[index.row()]
        if not self._device_open(device):
            self._open_device(device)

    def open_any(self):
        logger.debug("Opening any camera!")
        device = self.devices[0]
        if not self._device_open(device):
            self._open_device(device)

