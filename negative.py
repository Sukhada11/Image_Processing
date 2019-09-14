
import cv2
import numpy as np
from matplotlib import pyplot as plt
def rgb2gray(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            new[i,j]=((0.299 * frame[i,j,0]) + (0.587 * frame[i,j,1])+ (0.114 * frame[i,j,2]))/3
    
def negative(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
                new[i,j]=255-1-frame[i,j]

    
frame =  cv2.imread('color.jpg')
new = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)
neg = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)

rgb2gray(frame,new)
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cv2.imshow('frame',gray)
negative(new,neg)
plt.imshow(neg, cmap = plt.get_cmap('gray'))
plt.show()
