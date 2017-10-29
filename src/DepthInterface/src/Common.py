import os
import time
import logging
from logging.config import dictConfig

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
              'level': logging.DEBUG},
        '''
        'info_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.INFO,
            'formatter': 'simple',
            'filename': 'info.log',
            'maxBytes': 10485760,
            'backupCount': 20,
            'encoding': 'utf8'},
        'error_file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': logging.ERROR,
            'formatter': 'simple',
            'filename': 'errors.log',
            'maxBytes': 10485760,
            'backupCount': 20,
            'encoding': 'utf8'}
        '''
        },
    loggers = {
        'OpenGL.GL.shaders': {
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
