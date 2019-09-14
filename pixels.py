import cv2
import numpy as np
from matplotlib import pyplot as plt

frame =  cv2.imread('screen-b.png')
print()
#cv2.imshow('ty',frame)
image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split((hsv))
#print(image)
final = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
#cv2.imshow('final', frame)
cv2.waitKey(0) 


cv2.destroyAllWindows()