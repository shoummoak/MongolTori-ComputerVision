#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:39:20 2020

@author: raisin
"""

import numpy as np
import cv2
import cv2.aruco as aruco

# we will not use a built-in dictionary, but we could
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# define an empty custom dictionary with 
aruco_dict = aruco.custom_dictionary(0, 7, 1)
# add empty bytesList array to fill with 3 markers later
aruco_dict.bytesList = np.empty(shape = (3, 7, 4), dtype = np.uint8)

# add new marker(s)
mybits = np.array([[0]*7, [0,1,1,0,1,1,0], [0,1,1,0,1,1,0], [0,1,0,1,0,1,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0]*7], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0]*7, [0,1,1,0,1,1,0], [0,1,1,0,1,1,0], [0,1,0,1,0,1,0], [0,1,0,1,1,0,0], [0,1,0,1,1,0,0], [0]*7 ], dtype = np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0]*7, [0,1,0,1,1,1,0], [0,0,1,0,1,1,0], [0,1,1,1,0,0,0], [0,0,1,0,1,1,0], [0,0,1,1,1,1,0], [0]*7], dtype = np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(mybits)

# save marker images
for i in range(len(aruco_dict.bytesList)):
    cv2.imwrite("AR_test_images/custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))