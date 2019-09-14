import cv2
import numpy as np
from matplotlib import pyplot as plt

frame =  cv2.imread('color.jpg')
new = np.zeros((frame.shape[0],frame.shape[1]),dtype=np.int8)
for i in range(frame.shape[0]):
    for j in range(frame.shape[1]):
           new[i,j]=((0.299 * frame[i,j,0]) + (0.587 * frame[i,j,1])+ (0.114 * frame[i,j,2]))/3
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#plt.imshow(new, cmap = plt.get_cmap('gray'))

plt.imshow(new, cmap = plt.get_cmap('gray'))
plt.show()
#cv2.imshow('ray',new)
#cv2.imshow('org',image)
#cv2.waitKey(0) 


#cv2.destroyAllWindows()

    
