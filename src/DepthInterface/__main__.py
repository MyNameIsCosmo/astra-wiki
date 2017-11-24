__version__ = "0.1.0"
__author__ = "Cosmo Borsky"
__copyright__ = (
        "Copyright (c) 2017 Cosmo Borsky\n"
        "\n"
        "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
        "of this software and associated documentation files (the \"Software\"), to deal\n"
        "in the Software without restriction, including without limitation the rights\n"
        "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
        "copies of the Software, and to permit persons to whom the Software is\n"
        "furnished to do so, subject to the following conditions:\n"
        "\n"
        "The above copyright notice and this permission notice shall be included in all\n"
        "copies or substantial portions of the Software.\n"
        "\n"
        "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
        "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
        "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
        "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
        "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
        "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
        "SOFTWARE."
)
__license__ = "BSD License"
__date__ = "Nov 23, 2017"
__maintainer__ = "Cosmo Borsky"
__email__ = "me@cosmoborsky.com"
__status__ = "Development"

'''
TODO: 
    Camera intrinsics
    Camera position offset in 3d space
    Pop-up for grid properties
    Pop-up for camera properties
'''

import sys
import os
if __name__ == "__main__" and (__package__ is None or __package__==''):
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    import DepthInterface
    __package__ = "DepthInterface"
import signal
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Interface")
    parser.add_argument('--3d', action='store_true', help="Enable 3d point cloud display")
    parser.add_argument('--openni', default="use_default", help="Point to an openni Redist directory")
    parser.add_argument('--open-any', action="store_true", help="Skip device list and open any device")
    parser.add_argument('--debug', action="store_true", help="Print debug information")
    parser.add_argument('--version', action='store_true', help="Print version then exit")
    args = vars(parser.parse_args())

    #TODO: set logger to debug level here
    if args['version']:
        print("\nDepth Interface Version = " + __version__ + "\n")
        sys.exit(0)

# Load these after argparse
from .src.Common import *
from .src import QtDepthInterface

def run(args):

    app = QtGui.QApplication(sys.argv)

    main = QtDepthInterface.DepthInterface(args)

    app.aboutToQuit.connect(main._destruct)
    signal.signal(signal.SIGINT, main._destruct) #main._close if you want a pop-up

    # Let the interpreter run every 500ms
    # https://stackoverflow.com/a/4939113
    timer = QtCore.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    main.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    if args['openni']== "use_default":
        if sys.platform == "linux" or sys.platform == "linux2":
            args['openni'] = "OpenNI-Linux-x64-2.3/Redist"
        elif sys.platform == "win32":
            args['openni'] = "OpenNI-Windows-x64-2.3\\Redist"

    run(args)
