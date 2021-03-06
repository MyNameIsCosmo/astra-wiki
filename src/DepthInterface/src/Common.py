import os
import sys
import time
import logging
import ctypes
import cv2
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
from logging.config import dictConfig
from openni import openni2
from openni import _openni2 as c_api

platform = sys.platform

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

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
    loggers = {
        'OpenGL.GL.shaders': {
            'level': logging.ERROR,
            'handlers': ['console'],
            'propogate': False},
        'OpenGL.formathandler': {
            'level': logging.ERROR,
            'handlers': ['console'],
            'propogate': False},
        'OpenGL.extensions': {
            'level': logging.ERROR,
            'handlers': ['console'],
            'propogate': False}
        },
    root = {
        'handlers': ['console'],
        'level': logging.DEBUG,
        },
)

dictConfig(logging_config)
logger = logging.getLogger()
