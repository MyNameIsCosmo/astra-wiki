#!/usr/bin/python

if __name__ == "__main__":
    import sys
    import cv2
    import time
    import logging
    import ctypes
    import numpy as np
    from logging.config import dictConfig
    from openni import openni2
    from openni import _openni2 as c_api
else:
    from .Common import *

logging_config = dict(
    version = 1,
    disable_existing_loggers = False, 
    formatters = {
        'simple': {'format':
              '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'}
        },
    handlers = {
        'console': {'class': 'logging.StreamHandler',
              'formatter': 'simple',
              'level': logging.DEBUG}
        },
    root = {
        'handlers': ['console'],
        'level': logging.DEBUG,
        },
)
dictConfig(logging_config)
logger = logging.getLogger()


'''
TODO:
    QT config interface
    QT device finder/selector
    Multiple device viewer
    OpenCV image in QT interface
    PCL integration
    QT GL 3d viewer
    QT VTK 3d viewer
'''

'''
OpenNI Options:
    IMAGE_REGISTRATION_DEPTH_TO_COLOR
    IMAGE_REGISTRATION_OFF
Sensor Options:
    SENSOR_COLOR
    SENSOR_DEPTH
    SENSOR_IR
Pixel Format Options:
    PIXEL_FORMAT_DEPTH_100_UM
    PIXEL_FORMAT_DEPTH_1_MM
    PIXEL_FORMAT_GRAY16
    PIXEL_FORMAT_GRAY8
    PIXEL_FORMAT_JPEG
    PIXEL_FORMAT_RGB888
    PIXEL_FORMAT_SHIFT_9_2
    PIXEL_FORMAT_SHIFT_9_3
    PIXEL_FORMAT_YUV422
    PIXEL_FORMAT_YUYV
OpenNI Functions:
    configure_logging(directory=None, severity=None, console=None)
    convert_depth_to_color(depthStream, colorStream, depthX, depthY, depthZ)
    convert_depth_to_world(depthStream, depthX, depthY, depthZ)
    convert_world_to_depth(depthStream, worldX, worldY, worldZ)
    get_bytes_per_pixel(format)
    get_log_filename()
    get_version()
    initialize(dll_directories=['.'])
    is_initialized()
    unload()
    wait_for_any_stream(streams, timeout=None)
OpenNI.device functions:
    get_device_info()
    get_sensor_info(SENSOR_TYPE)
    has_sensor(SENSOR_TYPE)
    create_stream(SENSOR_TYPE)
    is_image_registration_mode_supported(True/False)
    get_image_registration_mode()
    set_image_registration_mode(True/False)
    .depth_color_sync
'''

STREAM_NAMES = {1: "ir", 2: "color", 3: "depth"}

def openni_init(path="."):
    if path is None:
        path = "."
    if path:
        if not "Redist" in path:
            if "linux" in sys.platform:
                path = path.rstrip('/') + "/Redist"
            elif "win32" in sys.platform:
                path = path.rstrip('\\') + "\\Redist"
    try:
        if (not openni2.is_initialized()):
            logger.info("OpenNi2 is not Initialized! Initializing.")
            openni2.initialize(path)
        return True
    except Exception as e:
        logger.error(e)
        logger.warning("Openni path is: " + path)

    try:
        logger.warning("Resorting to standard openni2 initialization")
        openni2.initialize()
        return True
    except Exception as e:
        logger.fatal(e)

    return False


def openni_list(path=""):
    openni_init(path)
    pdevs = ctypes.POINTER(c_api.OniDeviceInfo)()
    count = ctypes.c_int()
    c_api.oniGetDeviceList(ctypes.byref(pdevs), ctypes.byref(count))
    devices = [(pdevs[i].uri, pdevs[i].vendor, pdevs[i].name) for i in range(count.value)]
    c_api.oniReleaseDeviceList(pdevs)
    
    return devices

class VideoMode():
    def __init__(self, oniVideoMode=None, pixelFormat=None, resolutionX=None, resolutionY=None, fps=None):
        #TODO: clean up, make this a better object
        if oniVideoMode is not None:
            fps = oniVideoMode.fps
            pixelFormat = oniVideoMode.pixelFormat
            resolutionX = oniVideoMode.resolutionX
            resolutionY = oniVideoMode.resolutionY

        self.fps = fps
        self.pixelFormat = pixelFormat
        self.resolutionX = resolutionX
        self.resolutionY = resolutionY

        #print("{} {}:{}@{}".format(pixelFormat, resolutionX, resolutionY, fps))

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return ('{} {}:{}@{}'.format(self.pixelFormat, self.resolutionX, self.resolutionY, self.fps))

    def __repr__(self):
        return ('{} {}:{}@{}'.format(repr(self.pixelFormat), repr(self.resolutionX), repr(self.resolutionY), repr(self.fps)))

    def video_mode(self):
        return openni2.VideoMode(pixelFormat = self.pixelFormat, resolutionX = self.resolutionX, resolutionY = self.resolutionY, fps = self.fps)

class OpenNIStream(openni2.VideoStream):
    #TODO: Handle different cameras (Kinect vs Astra)
    def __init__(self, device, sensor_type):
        openni2.VideoStream.__init__(self, device, sensor_type)
        self.sensor_type = sensor_type
        self.active = False
        self.ctype = None
        self.frame = None
        self.frame_data = None

        self.video_modes = list()
        for mode in self.get_sensor_info().videoModes:
            video_mode = VideoMode(mode)
            self.video_modes.append(video_mode)

        self.settings = openni2.CameraSettings(self)

        #self._set_video_mode(None)

    def _set_video_mode(self, video_mode=None):
        if video_mode is None:
            logger.debug('Setting video mode to default')
            if self.default_video_mode is not None:
                video_mode = self.default_video_mode
            else:
                video_mode = self.video_modes[0]
        logger.debug("First video mode: {}".format(self.video_modes[0]))
        logger.debug("Current video mode: {}".format(video_mode))
        logger.debug("Default video mode: {}".format(self.default_video_mode))
        if video_mode in self.video_modes:
            #VideoMode(pixelFormat = pixelFormat, resolutionX = width, resolutionY = height, fps = fps)
            self.set_video_mode(video_mode.video_mode())
        else:
            logger.error('Video Mode not valid for {0}\n{1}'.format(self.sensor_type, video_mode))
            return False
        return True

    def _getData(self, ctype = None):
        ctype = ctype if (ctype) else self.ctype
        self.frame = self.read_frame()
        self.height = self.frame.height
        self.width = self.frame.width
        frame_data_buffer = self.frame.get_buffer_as(ctype)
        if (ctype is ctypes.c_ubyte):
            dtype = np.uint8
        else:
            # FIXME: Handle this better? map pixelFormat to np/ctype?
            dtype = np.uint16
        #self.frame_data = np.ndarray((self.height,self.width),dtype=dtype,buffer=frame_data_buffer)
        self.frame_data = np.frombuffer(frame_data_buffer, dtype=dtype)
        return self.frame_data

class OpenNIStream_Color(OpenNIStream):
    def __init__(self, device):
        self.default_video_mode = VideoMode(pixelFormat=openni2.PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30)
        OpenNIStream.__init__(self, device, openni2.SENSOR_COLOR)
        self.ctype = ctypes.c_uint8

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (self.height, self.width, 3) #reshape
        # self.frame_data = self.frame_data[...,::-1] #BGR to RGB
        return self.frame_data

class OpenNIStream_Depth(OpenNIStream):
    def __init__(self, device):
        self.default_video_mode = VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps=30)
        OpenNIStream.__init__(self, device, openni2.SENSOR_DEPTH)
        self.ctype = ctypes.c_uint16

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (self.height, self.width)
        return self.frame_data

class OpenNIStream_IR(OpenNIStream):
    def __init__(self, device):
        self.default_video_mode = VideoMode(pixelFormat=openni2.PIXEL_FORMAT_GRAY16, resolutionX=640, resolutionY=480, fps=30)
        OpenNIStream.__init__(self, device, openni2.SENSOR_IR)
        self.ctype = ctypes.c_uint16

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (self.height, self.width) #reshape
        return self.frame_data

class OpenNIDevice(openni2.Device):
    def __init__(self, uri=None, mode=None, path=None):
        openni_init(path)
        openni2.configure_logging(severity=0, console=False)
        openni2.Device.__init__(self, uri)
        self.serial = self.get_property(c_api.ONI_DEVICE_PROPERTY_SERIAL_NUMBER, (ctypes.c_char * 16)).value
        self.stream = {'color': OpenNIStream_Color(self),
                       'depth': OpenNIStream_Depth(self),
                       'ir': OpenNIStream_IR(self)}

    def stop(self):
        for s in self.stream:
            self.stream[s].stop()
        #openni2.unload()

    def open_stream(self, stream_type, width=None, height=None, fps=None, pixelFormat=None):
        try:
            if (not self.has_sensor(stream_type)):
                logger.error("Device does not have stream type of {}".format(stream_type))
                return False
            stream_name = STREAM_NAMES[stream_type.value]
            if self.stream[stream_name].active:
                logger.error("{} stream already active!".format(stream_name))
            self.stream[stream_name].start()
            video_mode = None
            if width is not None and height is not None and fps is not None and pixelFormat is not None:
                video_mode = openni2.VideoMode(pixelFormat=pixelFormat, resolutionX=width, resolutionY=height, fps=fps)
            self.stream[stream_name]._set_video_mode(video_mode)
            self.stream[stream_name].active = True
        except Exception as e:
            logger.error('Failed to open stream', exc_info=True)
            return False
        return True

    def open_stream_color(self, width=None, height=None, fps=None, pixelFormat=None):
        return self.open_stream(openni2.SENSOR_COLOR, width, height, fps, pixelFormat)

    def open_stream_depth(self, width=None, height=None, fps=None, pixelFormat=None):
        return self.open_stream(openni2.SENSOR_DEPTH, width, height, fps, pixelFormat)
        
    def open_stream_ir(self, width=None, height=None, fps=None, pixelFormat=None):
        return self.open_stream(openni2.SENSOR_IR, width, height, fps, pixelFormat)

    def get_frame(self, stream_type):
        '''
        frame_type = SENSOR_IR, SENSOR_COLOR, SENSOR_DEPTH
        '''
        try:
            if (not self.has_sensor(stream_type)):
                logger.warning("Device does not have stream type of {}".format(stream_type))
                return False

            stream_name = STREAM_NAMES[stream_type.value]
            if not self.stream[stream_name].active:
                logger.warning("{} stream not active!".format(stream_name))
                return False

            return self.stream[stream_name].getData()
        except Exception as e:
            logger.error("Failed to get frame", exc_info=True)
        return False

    def get_frame_color(self):
        return self.get_frame(openni2.SENSOR_COLOR)

    def get_frame_depth(self):
        return self.get_frame(openni2.SENSOR_DEPTH)

    def get_frame_ir(self):
        return self.get_frame(openni2.SENSOR_IR)


if __name__ == "__main__":
    device = OpenNIDevice()

    # FIXME: IR doesn't work?
    show_depth = False
    show_color = True
    show_ir = False

    if show_depth:
        device.open_stream_depth()
        cv2.namedWindow("Depth Image")
    if show_color:
        device.open_stream_color()
        cv2.namedWindow("Color Image")
    if show_ir:
        device.open_stream_ir()
        cv2.namedWindow("IR Image")

    if not show_ir and not show_color and not show_depth:
        logger.error("No streams enabled! Exiting")
        sys.exit(0)
    while True:
        if show_depth:
            depth_img = device.get_frame_depth()
            depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))
            cv2.imshow("Depth Image", depth_img)

        if show_color:
            color_img = device.get_frame_color()
            color_img = color_img[...,::-1] #BGR to RGB
            cv2.imshow("Color Image", color_img)

        if show_ir:
            ir_img = device.get_frame_ir()
            ir_img = cv2.convertScaleAbs(ir_img, alpha=(255.0/65535.0))
            cv2.imshow("IR Image", ir_img)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
            device.stop()
            break
    openni2.unload()
    cv2.destroyAllWindows()

