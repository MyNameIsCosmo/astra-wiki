.. _chapter_installation:

Installation
============

Linux
*****

Adapted from the Readme file provided by Orbbec.

.. NOTE::
  - For user with ARM based development board:
  - With CPU Structure older than Cortex A17, use OpenNI-Linux-Arm-2.3 Nofilter.tar for better performance.


- There are two zip files, one is for a 32bit machine and the other for a 64bit machine.

Let's choose 64bit (x64) and install the driver using an example as follows:


- To run visual samples(e.g., SimpleViewer), you will need ``freeglut3 header and libaries``, please install

::

    $ sudo apt-get install build-essential freeglut3 freeglut3-dev


Check the udev version, Orbbec Driver needs ``libudev.so.1``

If you can't find it then make a symbolic link from ``libudev.so.x.x``, which is usually located at ``/lib/x86_64-linux-gnu`` or ``/lib/i386-linux-gnu``

.. NOTE::
  Type the following commands, only if you could not find the right udev version.

::

    $ ldconfig -p | grep libudev.so.1
    $ cd /lib/x86_64-linux-gnu
    $ sudo ln -s libudev.so.x.x.x libudev.so.1

Next, let's install the OpenNI 2 driver.

- Download the tgz(or zip) file to directory (e.g., /home) and unzip it.

::

    $ unzip OpenNI-Linux-x64-2.3.zip
    $ cd OpenNI-Linux-x64-2.3


- Run ``install.sh`` to generate OpenNIDevEnvironment, which contains OpenNI development environment 

::

    $ sudo chmod a+x install.sh)
    $ sudo ./install.sh

- Please replug the Orbbec Astra device for usb-register

- Add environment variables

::

    $ source OpenNIDevEnvironment

- Build sample(e.g., SimpleViewer)

::

    $ cd Samples/SimpleViewer
    $ make

- Run a sample

::

    $ cd Bin/x64-Release
    $ ./SimpleViewer

- You should now be able to see a GUI window showing the depth stream video. Here is an image showing the GUI window


.. image:: _static/test.png

.. NOTE::
  - If the Debian Jessie Lite is used for testing, it may require the following installation for properly start the viewer.

::

    $ sudo apt-get install libgl1-mesa-dri

