#!/usr/bin/env python

import os
from distutils.core import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='DepthInterface',
      version='0.1',
      description='OpenNI Depth Camera Interface',
      long_description=read('README.md'),
      author='Cosmo Borsky',
      author_email='me@cosmoborsky.com',
      license='BSD',
      packages=['DepthInterface'],
      #package_dir={'': 'src'}
     )
