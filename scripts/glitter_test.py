#!/usr/bin/env python

"""Viewer for a random point cloud.

@author: Stephan Wenger
@date: 2012-03-13
"""

# Our scene is going to rotate. The rotating modelview matrix is computed using
# the sine and cosine functions from the math module:
from numpy import sin, cos, pi

# <i>glitter</i> uses <a href="http://numpy.scipy.org/">numpy</a> for
# representation of array data. We will use numpy's <code>randn()</code>
# function to generate random point coordinates:
from numpy.random import randn

# We can usually import classes and functions contained in <i>glitter</i>
# submodules directly from glitter:
from glitter import VertexArray, State, get_default_program

# Modules with external dependencies other than numpy, such as platform
# dependent parts like methods for the generation of an OpenGL context,
# however, have to be imported from their respective submodules:
from glitter.contexts.glut import GlutWindow, main_loop, get_elapsed_time

# <h2>Main class</h2>

# We wrap all the OpenGL interaction in a class. The class will contain an
# <code>__init__()</code> method to set up all OpenGL objects, any required
# callback methods, as well as a <code>run()</code> method to trigger execution
# of the GLUT main loop.
class PointcloudRenderer(object):
    # <h3>Initialization</h3>

    # When a <code>PointcloudRenderer</code> instance is created, we need to
    # initialize a few OpenGL objects.
    def __init__(self):
        # First, we create a window; this also creates an OpenGL context.
        self.window = GlutWindow(double=True, multisample=True)

        # Then, we set the GLUT display callback function which will be defined later.
        self.window.display_callback = self.display

        # In the OpenGL core profile, there is no such thing as a "standard pipeline"
        # any more. We use the minimalistic <code>defaultpipeline</code> from the
        # <code>glitter.convenience</code> module to create a shader program instead:
        self.shader = get_default_program()

        # Here, we create a vertex array that contains buffers for two vertex array
        # input variables as well as an index array:
        n_points = 100000
        self.vao = VertexArray(randn(n_points, 3), randn(n_points, 3))

    # <h3>Callback functions</h3>

    # <h4>Display function</h4>

    # Here we define the display function. It will be called by GLUT whenever the
    # screen has to be redrawn.
    def display(self):
        # First we clear the default framebuffer:
        self.window.clear()
        
        # To draw the vertex array, we use:
        self.vao.draw()

        # After all rendering commands have been issued, we swap the back buffer to
        # the front, making the rendered image visible all at once:
        self.window.swap_buffers()

    # <h4>Timer function</h4>

    # The animation is controlled by a GLUT timer. The timer callback changes the
    # modelview matrix, schedules the next timer event, and causes a screen redraw:
    def timer(self):
        # We first get the elapsed time from GLUT using <code>get_elapsed_time()</code>:
        phi = 2 * pi * get_elapsed_time() / 20.0

        # We then set the <code>modelview_matrix</code> uniform variable of the
        # shader created in the initialization section simply by setting an
        # attribute:
        self.shader.modelview_matrix = ((cos(phi), 0, sin(phi), 0), (0, 1, 0, 0), (-sin(phi), 0, cos(phi), 0), (0, 0, 0, 5))

        # The following line schedules the next timer event to execute after ten milliseconds.
        self.window.add_timer(10, self.timer)

        # Finally, we tell GLUT to redraw the screen.
        self.window.post_redisplay()

    # <h3>Running</h3>
    
    # We will call the <code>run()</code> method later to run the OpenGL code.
    def run(self):
        # To start the animation, we call the timer once; all subsequent timer
        # calls will be scheduled by the timer function itself.
        self.timer()

        # The default program is bound by using a <code>with</code> statement:
        with self.shader:
            # The <code>State</code> class encapsulates state changes in the
            # context. For example, to enable depth testing and set the point size
            # to three for the duration of the following function call, we would
            # write:
            with State(depth_test=True, point_size=3):
                # With the shader bound, depth testing enabled, and the point size
                # set, we enter the GLUT main loop.
                main_loop()

        # When the main loop exits, control is handed back to the script.

# Finally, if this program is being run from the command line, we instanciate
# the main class and run it.
if __name__ == "__main__":
    PointcloudRenderer().run()


