# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:53:18 2020

@author: ggb
"""

import matplotlib
import numpy as np

cmap = matplotlib.cm.get_cmap('seismic')
#rgba = cmap(0.5)
idx = np.linspace(0,1,256)

print str(cmap(0)[3])

iCount = 0
for ii in idx:
    #print( np.array(cmap(ii+1)) - np.array(cmap(ii)) )
    print( "lut.at<cv::Vec3b>(0," + str(iCount) + ") = cv::Vec3b(" 
    + str(int(cmap(ii)[0]*255)) + ", "
    + str(int(cmap(ii)[1]*255)) + ", "
    + str(int(cmap(ii)[2]*255)) + ");")
    iCount += 1
    