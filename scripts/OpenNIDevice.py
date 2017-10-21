#!/usr/bin/python
import ctypes
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

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

class OpenNIStream(openni2.VideoStream):
    def __init__(self, device, sensor_type):
        openni2.VideoStream.__init__(self, device, sensor_type)
        self.active = False
        self.x = 0
        self.y = 0
        self.fps = 0
        self.ctype = None
        self.pixelFormat = None
        self.frame = None
        self.frame_data = None

    def setVideoMode(self, x, y, fps, pixelFormat):
        self.x = x
        self.y = y
        self.fps = fps
        self.pixelFormat = pixelFormat
        self.set_video_mode(openni2.VideoMode(pixelFormat = pixelFormat, resolutionX = x, resolutionY = y, fps = fps))
        return True

    def _getData(self, ctype = None):
        ctype = ctype if (ctype) else self.ctype
        self.frame = self.read_frame()
        frame_data_buffer = self.frame.get_buffer_as(ctype)
        if (ctype is ctypes.c_ubyte):
            dtype = np.uint8
        else:
            # FIXME: Handle this better? map pixelFormat to np/ctype?
            dtype = np.uint16
        #self.frame_data = np.ndarray((self.frame.height,self.frame.width),dtype=dtype,buffer=frame_data_buffer)
        self.frame_data = np.frombuffer(frame_data_buffer, dtype=dtype)
        return self.frame_data

class OpenNIStream_Color(OpenNIStream):
    def __init__(self, device):
        OpenNIStream.__init__(self, device, openni2.SENSOR_COLOR)
        self.x = 640
        self.y = 480
        self.fps = 30
        self.ctype = ctypes.c_uint8
        self.pixelFormat = openni2.PIXEL_FORMAT_RGB888

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (480, 640, 3) #reshape
        self.frame_data = self.frame_data[...,::-1] #bgr to rgb
        return self.frame_data

class OpenNIStream_Depth(OpenNIStream):
    def __init__(self, device):
        OpenNIStream.__init__(self, device, openni2.SENSOR_DEPTH)
        self.x = 640
        self.y = 480
        self.fps = 30
        self.ctype = ctypes.c_uint16
        self.pixelFormat = openni2.PIXEL_FORMAT_DEPTH_100_UM

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (480, 640)
        return self.frame_data

class OpenNIStream_IR(OpenNIStream):
    def __init__(self, device):
        OpenNIStream.__init__(self, device, openni2.SENSOR_IR)
        self.x = 640
        self.y = 480
        self.fps = 30
        self.ctype = ctypes.c_uint16
        self.pixelFormat = openni2.PIXEL_FORMAT_GRAY16

    def getData(self, ctype = None):
        self._getData(ctype)
        self.frame_data.shape = (480, 640, 3) #reshape
        return self.frame_data

class OpenNIDevice(openni2.Device):
    def __init__(self, uri=None, mode=None):
        if (not openni2.is_initialized()):
            openni2.initialize()
        #openni2.configure_logging(severity=0, console=True)
        openni2.Device.__init__(self, uri)
        self.stream = {'color': OpenNIStream_Color(self),
                       'depth': OpenNIStream_Depth(self),
                       'ir': OpenNIStream_IR(self)}

    def stop(self):
        for s in self.stream:
            self.stream[s].stop()
        openni2.unload()

    def open_stream(self, stream_type, x=640, y=480, fps=30, pixelFormat=None):
        try:
            if (not self.has_sensor(stream_type)):
                print "Device does not have stream type of {}".format(stream_type)
                return False
            stream_name = STREAM_NAMES[stream_type.value]
            if (not pixelFormat):
                pixelFormat = openni2.PIXEL_FORMAT_GRAY16
            if self.stream[stream_name].active:
                print "Stream already active!"
            self.stream[stream_name].start()
            self.stream[stream_name].setVideoMode(x, y, fps, pixelFormat)
            self.stream[stream_name].active = True
        except Exception as e:
            print e
            return False
        return True

    def open_stream_color(self, x=640, y=480, fps=30, pixelFormat = openni2.PIXEL_FORMAT_RGB888):
        return self.open_stream(openni2.SENSOR_COLOR, x, y, fps, pixelFormat)

    def open_stream_depth(self, x=640, y=480, fps=30, pixelFormat = openni2.PIXEL_FORMAT_DEPTH_100_UM):
        return self.open_stream(openni2.SENSOR_DEPTH, x, y, fps, pixelFormat)
        
    def open_stream_ir(self, x=640, y=480, fps=30, pixelFormat = openni2.PIXEL_FORMAT_GRAY16):
        return self.open_stream(openni2.SENSOR_IR, x, y, fps, pixelFormat)

    def get_frame(self, stream_type):
        '''
        frame_type = SENSOR_IR, SENSOR_COLOR, SENSOR_DEPTH
        '''
        try:
            if (not self.has_sensor(stream_type)):
                print "Device does not have stream type of {}".format(stream_type)
                return False

            stream_name = STREAM_NAMES[stream_type.value]
            if not self.stream[stream_name].active:
                print "{} stream not active!".format(stream_name)
                return False

            return self.stream[stream_name].getData()
        except Exception as e:
            print e
        return False

    def get_frame_color(self):
        return self.get_frame(openni2.SENSOR_COLOR)

    def get_frame_depth(self):
        return self.get_frame(openni2.SENSOR_DEPTH)

    def get_frame_ir(self):
        return self.get_frame(openni2.SENSOR_IR)


if __name__ == "__main__":
    device = OpenNIDevice()

    import cv2

    # FIXME: IR doesn't work?
    
    cv2.namedWindow("Depth Image")
    cv2.namedWindow("Color Image")
    #cv2.namedWindow("IR Image")

    device.open_stream_depth()
    device.open_stream_color()
    #device.open_stream_ir()

    while True:
        depth_img = device.get_frame_depth()
        depth_img = cv2.convertScaleAbs(depth_img, alpha=(255.0/65535.0))

        cv2.imshow("Depth Image", depth_img)

        color_img = device.get_frame_color()
        cv2.imshow("Color Image", color_img)

        #ir_img = device.get_frame_ir()
        #cv2.imshow("IR Image", ir_img)


        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x') or key == ord("c")):
            device.stop()
            break
    cv2.destroyAllWindows()

