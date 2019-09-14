
import cv2
import numpy as np
from matplotlib import pyplot as plt
def rgb2gray(frame,new):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):     
            new[i,j]=int( (0.299 * frame[i,j,2]) + (0.587 * frame[i,j,1])+ (0.114 * frame[i,j,0]) )#/3
    
def hist1(frame,array):
    for y in range(0,256):
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                if y == int(frame[i,j]) :
                    array[y]=(array[y]+1)
                   
    print(frame.shape[0]*frame.shape[1])
    
def pow(frame,new,gamma):
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
                new[i,j]=np.power((frame[i,j]),gamma)
frame =  cv2.imread('color.jpg')
print(frame)
gray2 = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.int64)
#gray1 = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.int64)
#array=np.zeros((256),dtype = np.int64)
array1=np.zeros((256),dtype = np.int64)
array2=np.zeros((256),dtype = np.int64)
array3=np.zeros((256),dtype = np.int64)
rgb2gray(frame,gray2)
#gray1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#print(gray1)
#print(gray2)
hist1(gray2,array1)
#hist1(gray1,array)
var1 = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.int64)
var2 = np.zeros((frame.shape[0],frame.shape[1]),dtype = np.int64)
pow(gray2,var1,0.7)
pow(gray2,var2,1.5)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(gray2, cmap = plt.get_cmap('gray'))
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(var1, cmap = plt.get_cmap('gray'))
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(var2, cmap = plt.get_cmap('gray'))
plt.show()



hist1(var1,array2)
hist1(var2,array3)
print(array2)
print(array3)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
n, bins, patches = plt.hist(array1,255, facecolor='blue', alpha=0.5)
ax1.set_xlabel('Pixel values')
ax1.set_ylabel('Frequency')

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
n, bins, patches = plt.hist(array2,255, facecolor='green', alpha=0.5)
ax1.set_xlabel('Pixel values')
ax1.set_ylabel('Frequency')

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
n, bins, patches = plt.hist(array3,255, facecolor='red', alpha=0.5)
ax1.set_xlabel('Pixel values')
ax1.set_ylabel('Frequency')

#fig2 = plt.figure()
#ax2 = fig2.add_subplot(1, 1, 1)
#n, bins, patches = plt.hist(array, 255, facecolor='green', alpha=0.5)
#ax2.set_xlabel('Pixel values')
#ax2.set_ylabel('Frequency')
plt.show()
cv2.waitKey(0) 


cv2.destroyAllWindows()