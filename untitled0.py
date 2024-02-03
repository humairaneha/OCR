# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:18:03 2024

@author: humai
"""

#from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
from utils import four_point_transform,order_points,get_euler_distance
import numpy as np
import imutils
import cv2
import pytesseract
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread('med2.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
padding = 20

# Calculate the new dimensions for the padded image
height, width, _ = image.shape
padded_height = height + 2 * padding
padded_width = width + 2 * padding

# Create a black canvas with the new dimensions
padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)

# Copy the original image onto the center of the padded canvas
padded_image[padding:padding+height, padding:padding+width, :] = image

# Replace the original image with the padded image
image = padded_image
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((5,5),np.uint8)
#gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations= 3)

gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 75, 200)
#edged = cv2.dilate(edged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
kernel = np.ones((9, 9), np.uint8)
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


# show the original image and the edge detected image
print("STEP 1: Edge Detection")
#cv2.imwrite("Image", image)
cv2.imwrite("Edged.jpg", edged)
# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour

contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 300]
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
epsilon = 0.03 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
print(len(approx))
# loop over the contours
# Ensure that we have exactly 4 points
if len(approx) > 4:
    # Use minAreaRect to get the minimum area rectangle
    sorted_points = sorted(approx, key=lambda x: cv2.contourArea(np.array([x])))

    
    approx = sorted_points[-4:]
    approx = np.array(approx)
    hull = cv2.convexHull(np.array(approx))

   # Check if the points form a convex hull
    is_convex_hull = cv2.isContourConvex(approx)

    if is_convex_hull:
       print("The points form a convex hull.")
    else:
       print("The points do not form a convex hull.")
       approx = cv2.convexHull(np.array(approx))
       rect = cv2.minAreaRect(contours[0])
    
    # Get the box points of the rectangle
       approx = cv2.boxPoints(rect)
       approx = np.intp(approx)
      
       bgdModel = np.zeros((1,65),np.float64)
       fgdModel = np.zeros((1,65),np.float64)
       x, y, w, h = cv2.boundingRect(np.array(approx))
       img=image[y:y+h,x:x+w]
       cv2.imwrite("plot.jpg",img)
       mask = np.zeros(img.shape[:2],np.uint8)
       rect = (x,y,w,h)
       cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
       mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
       img = img*mask2[:,:,np.newaxis]
       cv2.imwrite("Outline2.jpg", img)
    
    # Draw the rectangle on the image
       #approx= approx.reshape((-1, 1, 2))
    #cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
    # Draw the rectangle on the image
    #cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
cv2.imwrite("Outline.jpg", image)
screenCnt=approx
#applying perspective transform & threshold

print("approx",approx[0])
warped,rect = four_point_transform(image, screenCnt.reshape(4, 2))
print(rect)
# convert the warped image to grayscale, then threshold it
# to give it that 'black and white' paper effect


src_pts = np.array(rect, dtype=np.float32)
cv2.polylines(image, [src_pts.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

width = int(get_euler_distance(src_pts[0], src_pts[1]))
height = int(get_euler_distance(src_pts[0], src_pts[3]))

dst_pts = np.array([[0, 0],   [0, width],  [height,width], [height, 0]], dtype=np.float32)

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warp = cv2.warpPerspective(image, M, (int(height), int(width)),flags=cv2.INTER_CUBIC)

warp= cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warp, 11, offset = 10, method = "median")
#warp = (warp > T).astype("uint8") * 255
res=pytesseract.image_to_string(warp)
print(res)
# show the original and scanned images
print("STEP 3: Apply perspective transform")
cv2.imwrite("Original.jpg", warp)



