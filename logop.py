#log function is used to expand value of dark pixels and compress value of light pixels
import cv2
import numpy as np
from matplotlib import pyplot as plt
def rgb2gray(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            new[i,j]=((0.299 * frame[i,j,0]) + (0.587 * frame[i,j,1])+ (0.114 * frame[i,j,2]))/3
    
def log(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
                new[i,j]=np.log(frame[i,j]+1)

    
frame =  cv2.imread('color.jpg')
new = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)
res = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)

rgb2gray(frame,new)

log(new,res)
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',gray)
plt.imshow(res, cmap = plt.get_cmap('gray'))
plt.show()
