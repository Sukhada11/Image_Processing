from kernels import kernel
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.signal import correlate
from skimage.exposure import rescale_intensity

img = cv2.imread("checker.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('original',img)
K1 =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])    
K2 =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


S1=kernel(img,K1)
S2=kernel(img,K2)
S3=S2+S1
S3 = rescale_intensity(S3, in_range=(0, 255))
S3= (S3 * 255).astype("uint8")

cv2.imshow('sobel',S3)
cv2.waitKey(0) 
cv2.destroyAllWindows()
