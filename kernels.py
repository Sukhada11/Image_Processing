import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

def kernel(I,K):
 S=np.zeros((I.shape[0],I.shape[1]),dtype=np.float32)
 m=K.shape[0]
 n=K.shape[1]
 for i in range(I.shape[0]):
   for j in range(I.shape[1]):
     for a in range (-math.floor(m/2),math.floor(m/2)+1):
       for b in range (-math.floor(n/2),math.floor(n/2)+1):
        #print ("a ",a)
       # print("b " ,b)
        if i-a<0 or j-b<0 or i-a>=I.shape[0] or j-b>=I.shape[1] :
           w=-1
        else:
           w=I[i-a,j-b]
        #print("w: ",w)
        if (math.floor(m/2)+a)<0 or (math.floor(n/2)+b)<0 or (math.floor(m/2)+a)>=K.shape[0] or (math.floor(n/2)+b)>=K.shape[1] :
           q=-1
        else:
           q=K[math.floor(m/2)+a,math.floor(n/2)+b]
        #print("q: ",q)
        #print(w*q)
        S[i,j]=S[i,j]+w*q
 return S
def kernel1(image,kernel):
    
    
    output=np.zeros((image.shape[0],image.shape[1]),dtype=np.float32)
    pad = (kernel.shape[0] - 1) // 2
    print(pad)
    image_padded = np.zeros((image.shape[0] + 2*pad, image.shape[1] + 2*pad),dtype=np.float32)  
    image_padded[1:-1,1:-1] = image    
    print(image.shape)
   
    for y in range(pad, image.shape[1]+ pad):
        for x in range(pad, image.shape[0] + pad):

            roi = image_padded[x - pad:x + pad + 1,y - pad:y + pad + 1]
            k = (roi * kernel).sum()
            output[ x - pad,y - pad] = k
    return output

x=np.array([[1,2,3],[4,5,6],[7,8,9]])
#print(x[0:2]) 
#for i in range 
#1 1 1
#1 -8 1
#1 1 1   ----> blur
#-1 -1 -1
#-1  8  -1 --> sharpen
#-1 -1 -1
#sobel
#-1 0 1
#-2 0 2
#-1 0 1
 
#-1 -2 -1 
# 0  0  0
# 1  2  1
#priwitzz
#-1 0 1
#-1 0 1
#-1 0 1
#
#-1 -1 -1
# 0  0  0
# 1  1  1

if __name__ == "__main__":
    K1 =  1/8*(np.array([[1,1,1],[1,1,1],[1,1,1]]))
    K1 =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])    
    K2 =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    K2 =  np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    K3= np.array(([0,1,0],[1,-4,1],[0,1,0]))
    from scipy.signal import correlate
    from skimage.exposure import rescale_intensity

    img = cv2.imread("checks.jpg")
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    S1=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    S2=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)
    S3=np.zeros((img.shape[0],img.shape[1]),dtype=np.float32)


    S1=kernel(img,K1)
    S3=kernel1(img,K2)
    S2=S1+S3

    S2 = rescale_intensity(S2, in_range=(0, 255))
    S2 = (S2 * 255).astype("uint8")
    S1 = rescale_intensity(S1, in_range=(0, 255))
    S1 = (S1 * 255).astype("uint8")

    S3 = rescale_intensity(S3, in_range=(0, 255))
    S3 = (S3 * 255).astype("uint8")
    cv2.imshow('filter',S2)
    cv2.imshow('filterdoonowtf',S3)
#print(img.shape)
#kernel(img,K3,S1)
#kernel(img,K1,S2)
#grad = np.sqrt(np.square(S1) + np.square(S2))
#grad *= 255.0 / grad.max()
#pRINT(img)
#print(S2)
#S[i,j]=S[i,j]/m*n;
#print(S1)
    cv2.imshow('prev',S1)

#cv2.imshow('aft',img-S1)



    cv2.waitKey(0) 


    cv2.destroyAllWindows()