import cv2
import os
import pdb
import copy
import math
import numpy as np

img =cv2.imread("sample.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
width,height = img.shape


theta = 30
radians = float(theta*(math.pi/180))
cx,cy = int(img.shape[0]/2),int(img.shape[1]/2)
new_width = int((width)*abs(np.cos(radians))+(height)*abs(np.sin(radians)))
new_height = int((width)*abs(np.sin(radians))+(height)*abs(np.cos(radians)))
print(new_width,new_height)
xoffset, yoffset = int(( width- new_width ) / 2.0), int(( height- new_height) / 2.0)
new1 = np.zeros((new_width,new_height ), dtype=np.uint8)
for w in range(xoffset,new_width) :
    for h in range(yoffset,new_height) :
        new_w = int((w-cx)*abs(np.cos(radians))-(h-cy)*abs(np.sin(radians)))+cx
        new_h = int((w-cx)*abs(np.sin(radians))+(h-cy)*abs(np.cos(radians)))+cy

        if (new_w >= 0 and new_w < width ) and (new_h >= 0 and new_h < height ):

            new1[w-xoffset,h-yoffset] = img[new_w,new_h]

cv2.imshow("img",img)
cv2.imshow("new1",new1)

cv2.waitKey(0)