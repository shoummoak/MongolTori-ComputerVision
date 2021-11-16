import cv2 as cv
import numpy as np
import time

"""  
----------------------------------------------------------------------------
FUNCTIONS
----------------------------------------------------------------------------
"""

# blur_it()  pparams --> frame, filter_kernel_size, apply = True/False
# if apply == True, return blurred frame & True, else return a Nan & False

def blur_it(frame, kernel_size, apply):
    if apply:
        return  cv.GaussianBlur(frame, (kernel_size, kernel_size), 0), True
    else:
        return np.NaN, False
    

    
# canny_it()  params --> frame, sigma=0.33
# sigma = 1/3 is the standard ratio

def canny_it(frame, sigma = 0.33):
    
    # calculate the median of the graysscale frame
    med = np.median(frame)
    
    # find the upper and lower threshold values
    lower = int((1 - sigma) * med)
    upper = int((1 + sigma) * med)
    
    # return the Canny frame
    return cv.Canny(frame, lower, upper)

def print_all_cannys(*args):
    
    count = 0
    window = 'Canny ' 
    
    for arg in args:
        count += 1
        cv.imshow(window + str(count), arg)

"""  
----------------------------------------------------------------------------
CODE BODY 
----------------------------------------------------------------------------
"""

fps_limit = 0.1
start_time = time.time()
cam = cv.VideoCapture(0)

while True:
    
    # reading the frame from camera
    ret, frame = cam.read()
    
    
    
    frame2 = frame.copy()
    frame3 = frame.copy()
    frame4 = frame.copy()
    
    """ 1 --> blur to reduce noise"""
    frame_blur, frame_applied = blur_it(frame, kernel_size = 5, apply = True)
    
    """ 2 --> canny to filter edges"""
    
    # auto canny option
    frame_canny = canny_it(frame_blur)
    
    # tight canny option
#    frame_canny_tight = cv.Canny(frame_blur, 225, 250)
    
    """  3 --> contour detection """
    
    # find contours from canny frame
    # cv.RETR_EXTERNAL returns all the outer contours
    contours = cv.findContours(frame_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    
    for contour in contours:
        
        # approximate polygons for the closed contours
        perimeter = cv.arcLength(contour, True)
        approx_polygon = cv.approxPolyDP(contour, 0.08*perimeter, True)        
        
        # (only select contours) and (select contours with 4 vertices)
        if cv.isContourConvex(approx_polygon) and len(approx_polygon) == 4: 
                
            cv.drawContours(frame2, [approx_polygon], 0, (0, 0, 255), 2)
            
            # !! BOUNDING RECT
            x, y, w, h = cv.boundingRect(approx_polygon)
            width_height_ratio = float(h)/w
#                box_area = w*h
#                extent = float(cv.contourArea(approx_polygon)) / box_area                
            
            # (it's width should not exceed height) and (area > 800)
            if (width_height_ratio >= 0.9) and (cv.contourArea(approx_polygon) > 800):
                
                # drawing contour
                cv.drawContours(frame3, [approx_polygon], 0, (0, 255, 0), 2)
                # regular rect
                bounding_rect = cv.rectangle(frame4, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    
    
    """ DISPLAY WINDOWS """
    
#    cv.imshow('mask', mask)
    cv.imshow('4vertices', frame2) 
    cv.imshow('square', frame3) 
    cv.imshow('bound_rect', frame4) 
#    print_all_cannys(frame_canny)        
    
#        cv.drawContours(frame, [contour], 0, (255, 0, 0), 3)    

    """ END OF CODE """
    
    # press 'ESC' to terminate script
    key = cv.waitKey(1)
    if key == 27:
        break
    
cam.release()
cv.destroyAllWindows()
    