
import cv2
import numpy as np
from matplotlib import pyplot as plt
image =  cv2.imread('light.png')
#print(frame)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (1, 1), 0)
thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('light',thresh)
cv2.waitKey(0) 


cv2.destroyAllWindows()

