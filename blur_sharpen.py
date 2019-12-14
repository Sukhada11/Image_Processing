import cv2
import numpy as np
import math
img1 = cv2.imread('test.jpg')

img=np.array(([1, 2, 3,4],[4, 3, 2, 1],[1, 2, 3, 4],[1, 4, 3, 2]))
kernel = np.ones((3,3))
print(kernel)
kernel = kernel*0.21
print(kernel)
def blur(img,kernel):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dest = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    m=kernel.shape[0]
    n=kernel.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for a in range (-math.floor(m/2),math.floor(m/2)+1):
                for b in range (-math.floor(n/2),math.floor(n/2)+1):
        #print ("a ",a)
       # print("b " ,b)
                    if i-a<0 or j-b<0 or i-a>=img.shape[0] or j-b>=img.shape[1] :
                        w=0
                    else:
                        w=img[i-a,j-b]
        #print("w: ",w)
                    if (math.floor(m/2)+a)<0 or (math.floor(n/2)+b)<0 or (math.floor(m/2)+a)>=kernel.shape[0] or (math.floor(n/2)+b)>=kernel.shape[1] :
                        q=0
                    else:
                        q=kernel[math.floor(m/2)+a,math.floor(n/2)+b]
        #print("q: ",q)
        #print(w*q)
                    dest[i,j]=dest[i,j]+w*q
    print(dest)
    cv2.imshow('p',dest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
   
    
x=blur(img1,kernel)
