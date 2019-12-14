
import cv2
import numpy as np
from matplotlib import pyplot as plt


def rgb2gray(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            new[i,j]=int(((299 * frame[i,j,2]) + (587 * frame[i,j,1])+ (114 * frame[i,j,0]))/1000)
    
def binthreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j]>=value):
                new[i,j]=int(255)
            else:
                new[i,j]=int(0)
                
                
def invbinthreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j]>value):
                new[i,j]=int(0)
            else:
                new[i,j]=int(255)
                
                
def invbinthreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j]>value):
                new[i,j]=int(0)
            else:
                new[i,j]=int(255)
          
          
def truncthreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j] > value):
                new[i,j] = value
            else:
                new[i,j] = frame[i,j]
                
                
def zerothreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j] < value):
                new[i,j] = int(0)
            else:
                new[i,j] = frame[i,j]
                
                
def invzerothreshold(frame,new,value):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if (frame[i,j] > value):
                new[i,j]=int(0)
            else:
                new[i,j] = frame[i,j]
                
                
                
frame =  cv2.imread('rainbow.png')
print(frame.shape)

new = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
thresh = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
thresh1 = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
thresh2 = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
thresh3 = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
thresh4 = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.uint8)
rgb2gray(frame,new)
print(thresh.shape)
cv2.imshow('gry',new)
print(new)
#plt.imshow(new)
binthreshold(new,thresh,127)
invbinthreshold(new,thresh1,127)
truncthreshold(new,thresh2,127)
zerothreshold(new,thresh3,127)
invzerothreshold(new,thresh4,127)
#thresh = np.expand_dims(thresh, axis=-1)
print(thresh.shape)
cv2.imshow('frame',thresh)
cv2.imshow('frame1',thresh1)
cv2.imshow('frame2',thresh2)
cv2.imshow('frame3',thresh3)
cv2.imshow('fram4',thresh4)
plt.show()
cv2.waitKey(0)