from kernels import kernel
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.signal import correlate
from skimage.exposure import rescale_intensity

img = cv2.imread("color.jpg")
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
K1 =  (np.array([[1,1,1],[1,-8,1],[1,1,1]]))
S1=kernel(img,K1)
S1 = rescale_intensity(S1, in_range=(0, 255))
S1= (S1 * 255).astype("uint8")
cv2.imshow('blurred',S1)
cv2.waitKey(0) 
cv2.destroyAllWindows()
