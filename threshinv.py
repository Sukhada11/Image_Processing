
import cv2
import numpy as np
from matplotlib import pyplot as plt
def rgb2gray(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            new[i,j]=((0.299 * frame[i,j,0]) + (0.587 * frame[i,j,1])+ (0.114 * frame[i,j,2]))/3
    
def threshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j]>=value):
                new[i,j]=int(0)
            else:
                new[i,j]=int(255)
    
frame =  cv2.imread('color.jpg')
new = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)
thresh = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)

rgb2gray(frame,new)
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',gray)
threshold(new,thresh,20)
plt.imshow(thresh, cmap = plt.get_cmap('gray'))
plt.show()