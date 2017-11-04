#!/usr/bin/evn python

import numpy as np
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import matplotlib.pyplot as plt


def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print 'Artist picked:', event.artist
    print '{} vertices picked'.format(len(ind))
    print 'Pick between vertices {} and {}'.format(min(ind), max(ind)+1)
    print 'x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse)
    print 'Data point:', x[ind[0]], y[ind[0]]
    print

def distance_from_plane(p, plane):
    p0 = np.array(plane[0]) 
    p1 = np.array(plane[len(plane)/2])
    p2 = np.array(plane[-1])

    # These two vectors are in the plane
    v1 = p2 - p0
    v2 = p1 - p0

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p2)

#    print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

#    print p0, p1, p2

    u = p1 - p0
    v = p2 - p0
    # vector normal to plane
    n = np.cross(u, v)
    n /= np.linalg.norm(n)

    p_ = p - p0
    dist_to_plane = np.dot(p_, n)
    p_normal = np.dot(p_, n) * n
    p_tangent = p_ - p_normal

    closest_point = p_tangent + p0
    coords = np.linalg.lstsq(np.column_stack((u, v)), p_tangent)[0]

    return dist_to_plane

def distance(point, event):
    assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

    # Project 3d data space to 2d data space
    x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
    # Convert 2d data space to 2d screen space
    x3, y3 = ax.transData.transform((x2, y2))

    return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)


def calcClosestDatapoint(X, event):
    distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
    return np.argmin(distances)


def annotatePlot(X, index):
    # If we have previously displayed another label, remove it first
    if hasattr(annotatePlot, 'label'):
	annotatePlot.label.remove()
    # Get data point from array of points X, at position index
    x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
    dist_to_plane = distance_from_plane(X[index],plane)
    annotatePlot.label = plt.annotate( "Value {}\nDist to Plane: {}".format(index, dist_to_plane),
	xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
	bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
	arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    fig.canvas.draw()


def onMouseMotion(event):
    closestIndex = calcClosestDatapoint(data, event)
    annotatePlot (data, closestIndex)

# some random 3-dim points

mean = np.array([0.0,0.0,0.0])
cov = np.array([[1.0,-0.5,0.8], [-0.5,1.1,0.0], [0.8,0.0,1.0]])
data = np.random.multivariate_normal(mean, cov, 8)

print data
print np.meshgrid(data[:,0], data[:,1], data[:,2])

# some 3-dim points
#x = [1.2, 1.3, 1.6, 2.5, 2.3, 2.8]
#y = [167.0, 180.3, 177.8,160.4,179.6, 154.3]
#z = [-0.3, -0.8, -0.75, -1.21, -1.65, -0.68]
#data = np.c_[x,y,z]

#print data.shape

# regular grid covering the domain of the data
#X,Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 5), np.linspace(mn[1], mx[1], 5))
XX = X.flatten()
YY = Y.flatten()

order = 2    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = np.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
    print X.shape
    print Y.shape
    print Z.shape

    z = C[0]*data[:,0] + C[1]*data[:,1] + C[2]

    print('The equation is z= {0}x + {1}y + {2}'.format(C[0], C[1], C[2]))
    
    #plane = np.c_[X[0], Y[0], Z[:,0]]
    plane = np.c_[data[:,0], data[:,1], z] #get a higher resolution plane per data point for exact depth accuracy

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = np.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    #Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

    #z = C[0]*data[:,0] + C[1]*data[:,1] + C[2]
    Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]

    print('The equation is z= {}x^2 + {}y^2 + {}xy {}x + {}y + {}'.format(C[4], C[5], C[3], C[1], C[2], C[0]))
    
    #plane = np.c_[X[0], Y[0], Z[:,0]]
    data_x = data[:,0]
    data_y = data[:,1]
    z = C[4]*data_x**2. + C[5]*data_y**2. + C[3]*data_x*data_y + C[1]*data_x + C[2]*data_y + C[0]
    plane = np.c_[data[:,0], data[:,1], z] #get a higher resolution plane per data point for exact depth accuracy

#print C

#print data.shape
#print plane.shape

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('X')
plt.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion
plt.show()
