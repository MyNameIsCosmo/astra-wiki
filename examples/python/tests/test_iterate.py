import numpy as np

x = [np.nan, np.nan, np.nan, .1, .2, .3, .4, .5, .6, np.nan, .8, np.nan, np.nan, 1.1, np.nan]
y = [np.nan, np.nan, .11, .22, .33, .44, .55, .66, np.nan, .87, np.nan, np.nan, 1.11, np.nan, np.nan]
z = [np.nan, .111, .222, .333, .444, .555, .666, np.nan, .888, np.nan, np.nan, 1.111, np.nan, np.nan, np.nan]

x = np.array(x)
y = np.array(y)
z = np.array(z)

print x
print y
print z
print ""

data = np.c_[x,y,z]
print data
print data[np.isfinite(data).all(axis=1)]
print data[np.isfinite(data).any(axis=1)]
print ""

print data[:,0]
print data[:,1]
print data.shape
print data[:,0].shape
