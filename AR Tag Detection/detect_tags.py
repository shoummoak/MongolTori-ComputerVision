# Program to create custom ArUco dictionary using OpenCV and detect markers using webcam
# original code from: http://www.philipzucker.com/aruco-in-opencv/
# Modified by Iyad Aldaqre
# 12.07.2019

import numpy as np
import cv2
import cv2.aruco as aruco

# we will not use a built-in dictionary, but we could
# aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

# define an empty custom dictionary with 
aruco_dict = aruco.custom_dictionary(0, 7, 1)
# add empty bytesList array to fill with 3 markers later
aruco_dict.bytesList = np.empty(shape = (3, 7, 4), dtype = np.uint8)

# add custom AR tags to detect
mybits = np.array([[0]*7, [0,1,1,0,1,1,0], [0,1,1,0,1,1,0], [0,1,0,1,0,1,0], [0,0,1,1,1,0,0], [0,0,1,1,1,0,0], [0]*7], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0]*7, [0,1,1,0,1,1,0], [0,1,1,0,1,1,0], [0,1,0,1,0,1,0], [0,1,0,1,1,0,0], [0,1,0,1,1,0,0], [0]*7 ], dtype = np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(mybits)
mybits = np.array([[0]*7, [0,1,0,1,1,1,0], [0,0,1,0,1,1,0], [0,1,1,1,0,0,0], [0,0,1,0,1,1,0], [0,0,1,1,1,1,0], [0]*7], dtype = np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(mybits)

# open video capture from (first) webcam
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        #lists of ids and the corners beloning to each id
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict)
        # draw markers on farme
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        # resize frame to show even on smaller screens
        frame = cv2.resize(frame, None, fx = 1.6, fy = 1.5)
        # Display the resulting frame
        cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()