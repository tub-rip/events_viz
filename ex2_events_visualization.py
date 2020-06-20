# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:57:35 2020

Read and write event data.

Visualization of event data in 3D
- with or without polarity
- change of viewpoint and movie creation

Conversion of event data into frames (images):
- histograms of events
- thresholded (1 f-stop)
- brightness increment images
- time surfaces: exponential decay or average time
With polarity on the same representation or split by polarity


TO DO:

Voxel grid
- temporal interpolation ("linear")

Evolvig set of points on the image plane?
Movie of the events (with markers such as +,.)

I think this is more suitable for exercise 3 (image reconstruction?)
Take two frames, and all the events between the two frames
Compute the brightness increment image from the events.
Compute a prediction of the 2nd frame using the 1st frame, the brightness 
increment image and the estimated contrast threshold?
Or compute the contrast threshold that minimizes the error between 
the prediction and the ground truth.

Write it nicely in utility functions

@author: ggb
"""

import os
#import cv2
import numpy as np
from matplotlib import pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

plt.close('all')

"""
filename = 'slider_depth/images.txt'
img_timestamps = open(filename, "r")

for line in img_timestamps.readlines():
    print(line)
    
img_timestamps.close()
"""

# %% Read a file of events and write another file with a subset of them
filename_sub = 'slider_depth/events_chunk.txt'
"""
events_raw = open('slider_depth/events.txt', "r")
events_sub = open(filename_sub, "w")
# t, x, y, pol

for k in range(50000):
    line = events_raw.readline()
    #print(line)
    events_sub.write(line)
    
events_raw.close()
events_sub.close()
"""

# %% Read file with a subset of events
def extract_data(filename):
    infile = open(filename, 'r')
    timestamp = []
    x = []
    y = []
    pol = []
    for line in infile:
        words = line.split()
        # words[0]: t, words[1]: x
        timestamp.append(float(words[0]))
        x.append(int(words[1]))
        y.append(int(words[2]))
        pol.append(int(words[3]))
    infile.close()
    return timestamp,x,y,pol
    
timestamp, x, y, pol = extract_data(filename_sub)


# %% Plot

# To get the size of the sensor using a grayscale frame (in case of a DAVIS)
# filename_frame = 'slider_depth/images/frame_00000000.png'
# img = cv2.imread(filename_frame, cv2.IMREAD_GRAYSCALE)
# print img.shape
# img = np.zeros(img.shape, np.int)

img_size = (180,240)

# Brightness incremet image (Balance of event polarities)
img = np.zeros(img_size, np.int)
num_events = 5000
print("numevents = ", num_events)
for i in range(num_events):
    #timestamp[i]
    img[y[i],x[i]] += (2*pol[i]-1)

fig = plt.figure()
fig.suptitle('Balance of event polarities')
#plt.imshow(img, cmap='gray')
maxabsval = np.amax(np.abs(img))
plt.imshow(img, cmap='seismic_r', clim=(-maxabsval,maxabsval))
plt.colorbar()
plt.show()


# %% Positive and negative events in separate images
img_pos = np.zeros(img_size, np.int)
img_neg = np.zeros(img_size, np.int)
for i in range(num_events):
    if (pol[i] > 0):
        img_pos[y[i],x[i]] += 1
    else:
        img_neg[y[i],x[i]] += 1

fig = plt.figure()
fig.suptitle('Histogram of positive events')
plt.imshow(img_pos, cmap='gray_r')
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Histogram of negative events')
plt.imshow(img_neg)
plt.colorbar()
plt.show()


# %% Thresholded representation

# Saturated singl: -1, 0, 1
# For example, given by the last event polarity at each pixel
img = np.zeros(img_size, np.int)
for i in range(num_events):
    img[y[i],x[i]] = (2*pol[i]-1)

fig = plt.figure()
fig.suptitle('Last event polarity per pixel')
plt.imshow(img, cmap='gray')
#plt.imshow(img, cmap='bwr')
plt.colorbar()
plt.show()


# %% Time surface (or time map, or SAE)
num_events = len(timestamp)
print("numevents = ", num_events)

img = np.zeros(img_size, np.float32)
t_ref = timestamp[-1] # time of the last event in the packet
tau = 0.03 # decay parameter (in seconds)
for i in range(num_events):
    img[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay). Both polarities')
plt.imshow(img)
plt.colorbar()
plt.show()


# %% Time surface (or time map, or SAE), separated by polarity
sae_pos = np.zeros(img_size, np.float32)
sae_neg = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae_pos[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae_neg[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay) of positive events')
plt.imshow(sae_pos)
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Time surface (exp decay) of negative events')
plt.imshow(sae_neg)
plt.colorbar()
plt.show()

# Using color (Red/blue) --> colormap seismic


# %% Time surface (or time map, or SAE), separated by polarity
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[y[i],x[i]] = -np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay). Both polarities, signed')
#plt.imshow(sae)
plt.imshow(sae, cmap='seismic')
plt.colorbar()
plt.show()


# %% Balance of 
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[y[i],x[i]] = sae[y[i],x[i]] + np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[y[i],x[i]] = sae[y[i],x[i]] - np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay). Balance of both polarities.')
#plt.imshow(sae)
maxabsval = np.amax(np.abs(sae))
plt.imshow(sae, cmap='seismic', clim=(-maxabsval,maxabsval))
plt.colorbar()
plt.show()

# %% Average timestamp per pixel
sae = np.zeros(img_size, np.float32)
count = np.zeros(img_size, np.int)
for i in range(num_events):
    sae[y[i],x[i]] += timestamp[i]
    count[y[i],x[i]] += 1
    
# Compute per-pixel average if count at the pixel is >1
count [count < 1] = 1
sae = sae / count

fig = plt.figure()
fig.suptitle('Average timestamps regardless of polarity')
plt.imshow(sae)
plt.colorbar()
plt.show()


# %% Average timestamp per pixel. Separate by polarity
sae_pos = np.zeros(img_size, np.float32)
sae_neg = np.zeros(img_size, np.float32)
count_pos = np.zeros(img_size, np.int)
count_neg = np.zeros(img_size, np.int)
for i in range(num_events):
    if (pol[i] > 0):
        sae_pos[y[i],x[i]] += timestamp[i]
        count_pos[y[i],x[i]] += 1
    else:
        sae_neg[y[i],x[i]] += timestamp[i]
        count_neg[y[i],x[i]] += 1
    
# Compute per-pixel average if count at the pixel is >1
count_pos [count_pos < 1] = 1;  sae_pos = sae_pos / count_pos
count_neg [count_neg < 1] = 1;  sae_neg = sae_neg / count_neg

fig = plt.figure()
fig.suptitle('Average timestamps of positive events')
plt.imshow(sae_pos)
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Average timestamps of negative events')
plt.imshow(sae_neg)
plt.colorbar()
plt.show()

# %% 3D plot (evemts and frames)
# time axis horizontally

m = 2000 # Number of points to plot

#plt.close('all')

# Plot without polarity
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal') # only works for time in Z axis
ax.scatter(x[:m], timestamp[:m], y[:m], marker='.', c='b')
ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix] ')
ax.view_init(azim=-90, elev=-180)
plt.show()

# Change viewpoint with the mouse, for example


# %% Plot each polarity with a different color (red / blue)
idx_pos = np.asarray(pol[:m]) > 0
idx_neg = np.logical_not(idx_pos)
xnp = np.asarray(x[:m])
ynp = np.asarray(y[:m])
tnp = np.asarray(timestamp[:m])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal') # only works for time in Z axis
ax.scatter(xnp[idx_pos], tnp[idx_pos], ynp[idx_pos], marker='.', c='b')
ax.scatter(xnp[idx_neg], tnp[idx_neg], ynp[idx_neg], marker='.', c='r')
ax.set(xlabel='x [pix]', ylabel='time [s]', zlabel='y [pix]')
ax.view_init(azim=-90, elev=-180)
plt.show()


# %% Transition: from viewpoint [-140,-60] to [-90,-90]
num_interp_viewpoints = 60
ele = np.linspace(-150,-180, num=num_interp_viewpoints)
azi = np.linspace( -50, -90, num=num_interp_viewpoints)

# Create directory to save images and then create a movie
dirName = 'tempDir'
if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")
    
for ii in xrange(0,num_interp_viewpoints):
    ax.view_init(azim=azi[ii], elev=ele[ii])
    plt.savefig(dirName + "/movie%04d.png" % ii)

# %% Create a movii using ffmpeg static build (https://johnvansickle.com/ffmpeg/)
# Video coding options, such as lossless: https://trac.ffmpeg.org/wiki/Encode/H.264
def createMovie():
    os.system("/home/ggb/Downloads/ffmpeg-4.0.1-64bit-static/ffmpeg -r 20 -i " 
    + dirName  + "/movie%04d.png -c:v libx264 -crf 0 -y movie.mp4")

#createMovie()


# %% Generate video, at constant number of events or constant time intervals


# %% Try to reason about the events
# How many per pixel per second? (average or histogram)


# %% Voxel grid

#plt.close('all')

# First, count histogram
# Input: x,y,timestamp,pol

# Just use the first m events

num_bins = 5

t_max = np.amax(np.asarray(timestamp[:m]))
t_min = np.amin(np.asarray(timestamp[:m]))
t_range = t_max - t_min
dt_cell = t_range / num_bins
t_edges = np.linspace(t_min,t_max,num_bins+1) # boundaries of the cells; not needed

# Compute the 3D histogram
# "Zero-th order or nearest neighbor voting"
hist3d = np.zeros(img.shape+(num_bins,), np.int)
for ii in xrange(m):
    idx_t = int( (timestamp[ii]-t_min) / dt_cell )
    if idx_t >= num_bins:
        idx_t = num_bins-1 # only one element (the last one)
    hist3d[y[ii],x[ii],idx_t] += 1

# Checks:
#print hist3d.shape
#print np.sum(hist3d) # This should equal the number of votes

# %% Using numpy function histogramdd
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd

# Specify bin edges in each dimension
bin_edges = (np.linspace(0,img_size[0],img_size[0]+1), 
             np.linspace(0,img_size[1],img_size[1]+1), t_edges)
yxt = np.transpose(np.array([y[:m], x[:m], timestamp[:m]]))
hist3dd, edges = np.histogramdd(yxt, bins=bin_edges)

# Checks
#print np.sum(hist3dd)
#print np.linalg.norm( hist3dd - hist3d)

"""
# Debugging. Plot error images
idx = 0
fig = plt.figure()
plt.imshow(hist3d[:,:,idx])
plt.colorbar()
plt.show()

fig = plt.figure()
plt.imshow(hist3dd[:,:,idx] - hist3d[:,:,idx])
plt.colorbar()
plt.show()
"""

# %% Plot
# Example: https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html#sphx-glr-gallery-mplot3d-voxels-rgb-py

# prepare some coordinates
r, g, b = np.indices((img_size[0]+1,img_size[1]+1,num_bins+1))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(r,g,b, hist3d)
ax.set(xlabel='y', ylabel='x', zlabel='time bin')
plt.show()

# It is smart: there is no need to swap the data to plot with reordered axes
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
#ax.view_init(azim=-90, elev=-180)
#ax.view_init(azim=-63, elev=-145)
plt.show()

# %%
colors = np.zeros(hist3d.shape + (3,))
colors[..., 0] = hist3d
colors[..., 1] = hist3d
colors[..., 2] = hist3d

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d, facecolors=colors)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
#ax.view_init(azim=-90, elev=-180)
ax.view_init(azim=-63, elev=-145)
plt.show()


# %% Compute interpolated histogram
hist3d_interp = np.zeros(img.shape+(num_bins,), np.float64)
for ii in xrange(m-1):
    tn = (timestamp[ii] - t_min) / dt_cell
    ti = int(tn)
    dt = tn - ti
    # Voting on two adjacent cells
    hist3d_interp[y[ii],x[ii],ti  ] += 1. - dt
    if ti < num_bins-1:
        hist3d_interp[y[ii],x[ii],ti+1] += dt

# Checks
print np.sum(hist3d_interp)
# Some votes are lost because of the missing last layer
print np.linalg.norm( hist3d - hist3d_interp)

# Debugging
"""
idx = 2
fig = plt.figure()
plt.imshow(hist3d_interp[:,:,idx])
plt.colorbar()
plt.show()

fig = plt.figure()
plt.imshow(hist3d[:,:,idx])
plt.colorbar()
plt.show()

fig = plt.figure()
plt.imshow(hist3d_interp[:,:,idx] - hist3d[:,:,idx])
plt.colorbar()
plt.show()
"""

# %% Plot voxel grid

colors = np.zeros(hist3d_interp.shape + (3,))
colors[..., 0] = hist3d_interp
colors[..., 1] = hist3d_interp
colors[..., 2] = hist3d_interp

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d_interp, facecolors=colors)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
#ax.view_init(azim=-90, elev=-180)
ax.view_init(azim=-63, elev=-145)
plt.show()



# %%

hist3d_interp_pol = np.zeros(img.shape+(num_bins,), np.float64)
for ii in xrange(m-1):
    tn = (timestamp[ii] - t_min) / dt_cell
    ti = int(tn)
    dt = tn - ti
    # Voting on two adjacent cells
    hist3d_interp_pol[y[ii],x[ii],ti  ] += (1. - dt) * (2*pol[ii]-1)
    if ti < num_bins-1:
        hist3d_interp_pol[y[ii],x[ii],ti+1] += dt * (2*pol[ii]-1)

# Checks
print np.sum(hist3d_interp)
# Some votes are lost because of the missing last layer
print np.linalg.norm( hist3d - hist3d_interp)

# %%
maxabsval = np.amax(np.abs(hist3d_interp_pol))
colors = np.zeros(hist3d_interp_pol.shape + (3,))
tmp = (hist3d_interp_pol + maxabsval)/(2*maxabsval)
colors[..., 0] = tmp
colors[..., 1] = tmp
colors[..., 2] = tmp

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d_interp_pol, facecolors=colors)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
#ax.view_init(azim=-90, elev=-180)
ax.view_init(azim=-63, elev=-145)
plt.show()
