import cv2
import numpy as np

# Load two images
img1 = cv2.imread('main.jpg')
img2 = cv2.imread('logo.jpg')
rows,cols,channels = img2.shape
rows1,cols1,channels1 = img1.shape
roi = img1[0:rows, cols1-cols:cols1 ]
#cv2.imshow('roi',roi)
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',img2gray)
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow('inv mask',mask_inv)
cv2.imshow('mask',mask)

img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
cv2.imshow('bg2',img1_bg)
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
#cv2.imshow('img2_fg',img2_fg)
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, cols1-cols:cols1 ] = dst

cv2.imshow('res',img1)
cv2.imwrite('wallpaper.jpg',img1)
#cv2.imshow('main',img1)
#cv2.imshow('logo',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()