# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:57:35 2020

Read and write event data.

Conversion of event data into frames (images, 2D):
- histograms of events
- thresholded (1 f-stop)
- brightness increment images
- time surfaces: exponential decay or average time
With polarity on the same representation or split by polarity

Visualization of event data in 3D
- with or without polarity
- change of viewpoint and movie creation

Write it nicely in utility functions

@author: ggb
"""

import os
import numpy as np
from matplotlib import pyplot as plt

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

plt.close('all')


# %% Read a file of events and write another file with a subset of them
filename_sub = 'slider_depth/events_chunk.txt'
"""
# This is how the file events_chunk.txt was generated from the events.txt file in the IJRR 2017 dataset
events_raw = open('slider_depth/events.txt', "r")
events_sub = open(filename_sub, "w")
# format: timestamp, x, y, polarity

for k in range(50000):
    line = events_raw.readline()
    #print(line)
    events_sub.write(line)
    
events_raw.close()
events_sub.close()
"""


# %% Read file with a subset of events
# Simple. There may be more efficient ways.
def extract_data(filename):
    infile = open(filename, 'r')
    timestamp = []
    x = []
    y = []
    pol = []
    for line in infile:
        words = line.split()
        timestamp.append(float(words[0]))
        x.append(int(words[1]))
        y.append(int(words[2]))
        pol.append(int(words[3]))
    infile.close()
    return timestamp,x,y,pol

# Call the function to read data    
timestamp, x, y, pol = extract_data(filename_sub)


# %% Sensor size

# Get the size of the sensor using a grayscale frame (in case of a DAVIS)
# filename_frame = 'slider_depth/images/frame_00000000.png'
# import cv2
# img = cv2.imread(filename_frame, cv2.IMREAD_GRAYSCALE)
# print img.shape
# img = np.zeros(img.shape, np.int)

# For this exercise, we just provide the sensor size (height, width)
img_size = (180,240)


# %% Brightness incremet image (Balance of event polarities)
num_events = 5000  # Number of events used
print("Brightness incremet image: numevents = ", num_events)

# Compute image by accumulating polarities.
img = np.zeros(img_size, np.int)
for i in range(num_events):
    # Need to convert the polarity bit from {0,1} to {-1,+1} and accumulate
    img[y[i],x[i]] += (2*pol[i]-1)

# Display the image in grayscale
fig = plt.figure()
fig.suptitle('Balance of event polarities')
maxabsval = np.amax(np.abs(img))
plt.imshow(img, cmap='gray', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()

# Same plot as above, but changing the color map
fig = plt.figure()
fig.suptitle('Balance of event polarities')
maxabsval = np.amax(np.abs(img))
plt.imshow(img, cmap='seismic_r', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


# %% 2D Histograms of events, split by polarity (positive and negative events in separate images)
img_pos = np.zeros(img_size, np.int)
img_neg = np.zeros(img_size, np.int)
for i in range(num_events):
    if (pol[i] > 0):
        img_pos[y[i],x[i]] += 1 # count events
    else:
        img_neg[y[i],x[i]] += 1

fig = plt.figure()
fig.suptitle('Histogram of positive events')
plt.imshow(img_pos)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Histogram of negative events')
plt.imshow(img_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


# %% Thresholded brightness increment image (Ternary image)

# What if we only use 3 values in the event accumulation image?
# Saturated signal: -1, 0, 1
# For example, store the polarity of the last event at each pixel
img = np.zeros(img_size, np.int)
for i in range(num_events):
    img[y[i],x[i]] = (2*pol[i]-1)  # no accumulation; overwrite the stored value

# Display the ternary image
fig = plt.figure()
fig.suptitle('Last event polarity per pixel')
plt.imshow(img, cmap='gray')
#plt.imshow(img, cmap='bwr')
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()




# _____________________________________________________________________________
# %% Time surface (or time map, or SAE="Surface of Active Events")
num_events = len(timestamp)
print("Time surface: numevents = ", num_events)

img = np.zeros(img_size, np.float32)
t_ref = timestamp[-1] # time of the last event in the packet
tau = 0.03 # decay parameter (in seconds)
for i in range(num_events):
    img[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay). Both polarities')
plt.imshow(img)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
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
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Time surface (exp decay) of negative events')
plt.imshow(sae_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


# %% Time surface (or time map, or SAE), using polarity as sign of the time map
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[y[i],x[i]] = np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[y[i],x[i]] = -np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay), using polarity as sign')
plt.imshow(sae, cmap='seismic') # using color (Red/blue)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


# %% "Balance of time surfaces"
# Accumulate exponential decays using polarity as sign of the time map
sae = np.zeros(img_size, np.float32)
for i in range(num_events):
    if (pol[i] > 0):
        sae[y[i],x[i]] += np.exp(-(t_ref-timestamp[i]) / tau)
    else:
        sae[y[i],x[i]] -= np.exp(-(t_ref-timestamp[i]) / tau)

fig = plt.figure()
fig.suptitle('Time surface (exp decay), balance of both polarities')
#plt.imshow(sae)
maxabsval = np.amax(np.abs(sae))
plt.imshow(sae, cmap='seismic', clim=(-maxabsval,maxabsval))
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()


# %% Average timestamp per pixel
sae = np.zeros(img_size, np.float32)
count = np.zeros(img_size, np.int)
for i in range(num_events):
    sae[y[i],x[i]] += timestamp[i]
    count[y[i],x[i]] += 1
    
# Compute per-pixel average if count at the pixel is >1
count [count < 1] = 1  # to avoid division by zero
sae = sae / count

fig = plt.figure()
fig.suptitle('Average timestamps regardless of polarity')
plt.imshow(sae)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
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
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()

fig = plt.figure()
fig.suptitle('Average timestamps of negative events')
plt.imshow(sae_neg)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.colorbar()
plt.show()




# _____________________________________________________________________________
# %% 3D plot 
# Time axis in horizontal position

m = 2000 # Number of points to plot
print("Space-time plot and movie: numevents = ", m)

# Plot without polarity
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.set_aspect('equal') # only works for time in Z axis
ax.scatter(x[:m], timestamp[:m], y[:m], marker='.', c='b')
ax.set_xlabel('x [pix]')
ax.set_ylabel('time [s]')
ax.set_zlabel('y [pix] ')
ax.view_init(azim=-90, elev=-180) # Change viewpoint with the mouse, for example
plt.show()


# %% Plot each polarity with a different color (red / blue)
idx_pos = np.asarray(pol[:m]) > 0
idx_neg = np.logical_not(idx_pos)
xnp = np.asarray(x[:m])
ynp = np.asarray(y[:m])
tnp = np.asarray(timestamp[:m])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xnp[idx_pos], tnp[idx_pos], ynp[idx_pos], marker='.', c='b')
ax.scatter(xnp[idx_neg], tnp[idx_neg], ynp[idx_neg], marker='.', c='r')
ax.set(xlabel='x [pix]', ylabel='time [s]', zlabel='y [pix]')
ax.view_init(azim=-90, elev=-180)
plt.show()


# %% Transition between two viewpoints
num_interp_viewpoints = 60 # number of interpolated viewpoints
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

# %% Create a movie using ffmpeg static build (https://johnvansickle.com/ffmpeg/)
# Video coding options, such as lossless: https://trac.ffmpeg.org/wiki/Encode/H.264
def createMovie():
    os.system("/home/ggb/Downloads/ffmpeg-4.2.2-i686-static/ffmpeg -r 20 -i " 
    + dirName  + "/movie%04d.png -c:v libx264 -crf 0 -y movie_new.mp4")

# Call the function to create the movie
createMovie()




# _____________________________________________________________________________
# %% Voxel grid

num_bins = 5
print("Number of time bins = ", num_bins)

t_max = np.amax(np.asarray(timestamp[:m]))
t_min = np.amin(np.asarray(timestamp[:m]))
t_range = t_max - t_min
dt_bin = t_range / num_bins # size of the time bins (bins)
t_edges = np.linspace(t_min,t_max,num_bins+1) # Boundaries of the bins

# Compute 3D histogram of events manually with a loop
# ("Zero-th order or nearest neighbor voting")
hist3d = np.zeros(img.shape+(num_bins,), np.int)
for ii in xrange(m):
    idx_t = int( (timestamp[ii]-t_min) / dt_bin )
    if idx_t >= num_bins:
        idx_t = num_bins-1 # only one element (the last one)
    hist3d[y[ii],x[ii],idx_t] += 1

# Checks:
print("hist3d")
print hist3d.shape
print np.sum(hist3d) # This should equal the number of votes

 
# %% Compute 3D histogram of events using numpy function histogramdd
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogramdd.html#numpy.histogramdd

# Specify bin edges in each dimension
bin_edges = (np.linspace(0,img_size[0],img_size[0]+1), 
             np.linspace(0,img_size[1],img_size[1]+1), t_edges)
yxt = np.transpose(np.array([y[:m], x[:m], timestamp[:m]]))
hist3dd, edges = np.histogramdd(yxt, bins=bin_edges)

# Checks
print("\nhist3dd")
print("min = ", np.min(hist3dd))
print("max = ", np.max(hist3dd))
print np.sum(hist3dd)
print np.linalg.norm( hist3dd - hist3d) # Check: zero if both histograms are equal
print("Ratio of occupied bins = ", np.sum(hist3dd > 0) / float(np.prod(hist3dd.shape)) )


# Plot of the 3D histogram. Empty cells are transparent (not displayed)
# Example: https://matplotlib.org/3.1.1/gallery/mplot3d/voxels_rgb.html#sphx-glr-gallery-mplot3d-voxels-rgb-py

fig = plt.figure()
fig.suptitle('3D histogram (voxel grid), zero-th order voting')
ax = fig.gca(projection='3d')
# prepare some coordinates
r, g, b = np.indices((img_size[0]+1,img_size[1]+1,num_bins+1))
ax.voxels(g,b,r, hist3d) # No need to swap the data to plot with reordered axes
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-90, elev=-180) # edge-on, along time axis
#ax.view_init(azim=-63, elev=-145) # oblique viewpoint
plt.show()


# %% Compute interpolated 3D histogram (voxel grid)
hist3d_interp = np.zeros(img.shape+(num_bins,), np.float64)
for ii in xrange(m-1):
    tn = (timestamp[ii] - t_min) / dt_bin # normalized time, in [0,num_bins]
    ti = np.floor(tn-0.5) # index of the left bin
    dt = (tn-0.5) - ti    # delta fraction
    # Voting on two adjacent bins
    if ti >=0 :
        hist3d_interp[y[ii],x[ii],int(ti)  ] += 1. - dt
    if ti < num_bins-1 :
        hist3d_interp[y[ii],x[ii],int(ti)+1] += dt

# Checks
print("\nhist3d_interp")
print("min = ", np.min(hist3d_interp))
print("max = ", np.max(hist3d_interp))
print np.sum(hist3d_interp)
# Some votes are lost because of the missing last layer
print np.linalg.norm( hist3d - hist3d_interp)
print("Ratio of occupied bins = ", np.sum(hist3d_interp > 0) / float(np.prod(hist3d_interp.shape)) )

# Plot voxel grid
colors = np.zeros(hist3d_interp.shape + (3,))
tmp = hist3d_interp/np.amax(hist3d_interp) # normalize in [0,1]
colors[..., 0] = tmp
colors[..., 1] = tmp
colors[..., 2] = tmp

fig = plt.figure()
fig.suptitle('Interpolated 3D histogram (voxel grid)')
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d_interp, facecolors=colors)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-63, elev=-145)
plt.show()

# %% A different visualization viewpoint
ax.view_init(azim=-90, elev=-180) # edge-on, along time axis
plt.show()


# %% Compute interpolated 3D histogram (voxel grid) using polarity
hist3d_interp_pol = np.zeros(img.shape+(num_bins,), np.float64)
for ii in xrange(m-1):
    tn = (timestamp[ii] - t_min) / dt_bin # normalized time, in [0,num_bins]
    ti = np.floor(tn-0.5) # index of the left bin
    dt = (tn-0.5) - ti    # delta fraction
    # Voting on two adjacent bins
    if ti >=0 :
        hist3d_interp_pol[y[ii],x[ii],int(ti)  ] += (1. - dt) * (2*pol[ii]-1)
    if ti < num_bins-1 :
        hist3d_interp_pol[y[ii],x[ii],int(ti)+1] += dt * (2*pol[ii]-1)

# Checks
# Some votes are lost because of the missing last layer
print("\nhist3d_interp_pol")
print("min = ", np.min(hist3d_interp_pol))
print("max = ", np.max(hist3d_interp_pol))
print np.sum(np.abs(hist3d_interp_pol))
print("Ratio of occupied bins = ", np.sum(np.abs(hist3d_interp_pol) > 0) / float(np.prod(hist3d_interp_pol.shape)) )

# Plot interpolated voxel grid using polarity
# Normalize the symmetric range to [0,1]
maxabsval = np.amax(np.abs(hist3d_interp_pol))
colors = np.zeros(hist3d_interp_pol.shape + (3,))
tmp = (hist3d_interp_pol + maxabsval)/(2*maxabsval)
colors[..., 0] = tmp
colors[..., 1] = tmp
colors[..., 2] = tmp

fig = plt.figure()
fig.suptitle('Interpolated 3D histogram (voxel grid), including polarity')
ax = fig.gca(projection='3d')
ax.voxels(g,b,r, hist3d_interp_pol, facecolors=colors)
ax.set(xlabel='x', ylabel='time bin', zlabel='y')
ax.view_init(azim=-63, elev=-145)
plt.show()

# %% Better visualization viewpoint to see positive and negative edges
ax.view_init(azim=-90, elev=-180) # edge-on, along time axis
plt.show()
