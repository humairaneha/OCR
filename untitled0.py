# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:18:03 2024

@author: humai
"""

#from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import imutils
import cv2
# load the image and compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread('med2.jpg')
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
edged = cv2.Canny(gray, 75, 200)
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

# show the contour (outline) of the piece of paper
print("STEP 2: Find contours of paper")
cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
cv2.imwrite("Outline.jpg", image)
