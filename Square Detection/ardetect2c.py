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

fps_limit = 0.005
start_time = time.time()
cam = cv.VideoCapture(0)

while True:
    
    # reading the frame from camera
    ret, frame = cam.read()
    
    current_time = time.time()
    if (current_time - start_time) > fps_limit:
            

        frame2 = frame.copy()
        frame3 = frame.copy()
        frame4 = frame.copy()
        
        """ 1 --> blur to reduce noise"""
        frame_blur, frame_applied = blur_it(frame, kernel_size = 5, apply = True)
        """ 2 --> canny to filter edges"""
        # auto option
        frame_canny = canny_it(frame_blur)
        
        # wide option
        #    frame_canny_wide = cv.Canny(frame_blur, 10, 200)
        
        """  3 --> contour detection """
        
        # find contours from auto canny option
        # cv.RETR_EXTERNAL returns all the outer contours
        contours = cv.findContours(frame_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        
        tags = []
        
        for contour in contours:
            
            # approximate polygons for the closed contours
            perimeter = cv.arcLength(contour, True)
            approx_polygon = cv.approxPolyDP(contour, 0.08*perimeter, True)        
            
            # This filters out many of the open contours
            if cv.isContourConvex(approx_polygon):
                cv.drawContours(frame, [approx_polygon], 0, (255, 255, 0), 2)
                
                # pick polygons with four vertices
                if len(approx_polygon) == 4:
                    
                    cv.drawContours(frame2, [approx_polygon], 0, (0, 0, 255), 2)          
                    
                    # (it has a square shape) and (area > 200) and (actual_area / box_area > 0.65)
                    if (cv.contourArea(approx_polygon) > 1000):
                        
                        # MASK
                        mask = np.zeros(frame.shape[:2], np.uint8)
                        cv.drawContours(mask, [contour], 0, 255, -1)
                        pixelpoints = np.transpose(np.nonzero(mask))
                        mean_value = cv.mean(frame, mask = mask)
#                        print(mean_value)
                        
                        lower_blue = 70
                        upper_blue = 120
                        lower_green = 60
                        upper_green = 110
                        lower_red = 10
                        upper_red = 50
                        
                        if (
                            mean_value[0] >= lower_blue and mean_value[0] <= upper_blue and
                            mean_value[1] >= lower_green and mean_value[1] <= upper_green and
                            mean_value[2] >= lower_red and mean_value[2] <= upper_red
                            ):
                            tags.append(approx_polygon)
                            
        #                    # contour drawing                    
        #                    cv.drawContours(frame3, [approx_polygon], 0, (0, 255, 0), 2)
        #                    # regular rect
        #                    bounding_rect = cv.rectangle(frame4, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            for tag in tags:
                cv.drawContours(frame3, [tag], 0, (0, 255, 0), 2)               
        
        """ DISPLAY WINDOWS """
        
        cv.imshow('mask', mask)
        #    cv.imshow('4vertices', frame2) 
        cv.imshow('square', frame3) 
        #    cv.imshow('bound_rect', frame4) 
        #    print_all_cannys(frame_canny)
        start_time = time.time()
               
        """ END OF CODE """
        
    # code to terminate script
    key = cv.waitKey(1)
    if key == 27:
        break
    
cam.release()
cv.destroyAllWindows()
    