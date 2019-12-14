import cv2
import numpy as np
import math
img1 = cv2.imread('j.png')
img2 = cv2.imread('opening1.png')
img3 = cv2.imread('closing.png')

print(img2)
cv2.imshow('j',img2)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img=np.array(([1, 2, 3,4],[4, 3, 2, 1],[1, 2, 3, 4],[1, 4, 3, 2]))
kernel = np.ones((3,3))

#print(kernel)
def erode(img,kernel):

    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dest = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    m=kernel.shape[0]
    n=kernel.shape[1]
    flag=0
    c=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            flag=0
            for a in range (-math.floor(m/2),math.floor(m/2)+1):
                for b in range (-math.floor(n/2),math.floor(n/2)+1):
                    if i-a<0 or j-b<0 or i-a>=img.shape[0] or j-b>=img.shape[1] :
                        pass
                    else:
                        if img[i-a,j-b]!=255:#0
                            flag=1
            if flag==1 :
                dest[i,j]=0#255
            else :
                dest[i,j]=255#0
            flag=0
    return dest
    
def dilute(img,kernel):

    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dest = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    m=kernel.shape[0]
    n=kernel.shape[1]
    flag=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            for a in range (-math.floor(m/2),math.floor(m/2)+1):
                for b in range (-math.floor(n/2),math.floor(n/2)+1):
                    if i-a<0 or j-b<0 or i-a>=img.shape[0] or j-b>=img.shape[1] :
                        pass
                    else:
                        if img[i-a,j-b]!=0:
                            flag=1
            if flag==1 :
                dest[i,j]=255
            else :
                dest[i,j]=0
            flag=0

    return dest

def closing(img,kernel):
    x=dilute(img,kernel)
    y=erode(x,kernel)
    return y
    
def opening(img,kernel):
    x=erode(img,kernel)   
    y=dilute(x,kernel)
    return y
    
x=erode(img1,kernel)
y=dilute(img1,kernel)
img1=img1.astype(np.uint8)

a = opening(img2,kernel)
b = closing(img3,kernel)
cv2.imshow('before',img1)
cv2.imshow('erode',x)
cv2.imshow('dilute',y)
cv2.imshow('before0',img2)
cv2.imshow('open',a)
cv2.imshow('before1',img3)
cv2.imshow('close',b)
cv2.waitKey(0)
cv2.destroyAllWindows()