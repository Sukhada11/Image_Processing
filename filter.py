import cv2
import numpy as np
from matplotlib import pyplot as plt

#frame =  cv2.imread('blind2 (5).jpg')
frame =  cv2.imread('screen-b.png')
#frame =  cv2.imread('red.png')
print(frame.shape)

cv2.imshow('ty',frame)
img = frame
image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

'''
image = np.array(image, dtype=np.float)
image /= 255.0
cv2.imshow('img',img)


a_channel = np.ones(image.shape, dtype=np.float)/2.0
image1 = image*a_channel
cv2.imshow('img',image1)
'''
'''
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(image, (15, 15), 0)
red = img[:,:,2]
#print(red.shape)
red=np.expand_dims(red, axis=2)
#print(red.shape)

green = img[:,:,1]
green=np.expand_dims(green, axis=2)

blue = img[:,:,0]
#print(red.shape)
blue=np.expand_dims(blue, axis=2)
'''
zero = np.zeros([img.shape[0], img.shape[1],1], dtype=np.uint8)
one = np.ones([img.shape[0], img.shape[1],3], dtype=np.uint8)

#print(zero.shape)

#print(one)
#print(zero)
cv2.imshow("red",img)
#cv2.imshow("blue",blue)
#cv2.imshow("green",green)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
cv2.imshow("before",hsv)

print(frame)

#hsv = cv2.normalize(hsv,  hsv, 0, 255, cv2.NORM_MINMAX)
#cv2.imshow("after",hsv)
H,S,V= hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] 



h,s,v=cv2.split(hsv)
#print(h,s,v)
y =cv2.merge((h,s,v))

cv2.imshow("oky",y)
#print(y)



#-----Find L-channel minimum--
(minValH, maxValH, minLocH, maxLocH) = cv2.minMaxLoc(h)
(minValS, maxValS, minLocS, maxLocS) = cv2.minMaxLoc(s)
(minValV, maxValV, minLocV, maxLocV) = cv2.minMaxLoc(v)
#-----Subtract difference of L-channels--


#S = cv2.subtract(S,20)
#V = cv2.subtract(V,maxValV)
#print(h)
#print(minValS)
#print(minValH)
#print(maxValV)
#print(maxValS)
#print(maxValH)

#-----Merge the enhanced L-channel with the a and b channel--
final = cv2.merge((h,s,v))
final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
cv2.imshow("finalafterhue",final)
final = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
lower_red = np.array([0,15,0])
upper_red = np.array([30,255,255])
mask1 = cv2.inRange(final, lower_red, upper_red)
cv2.imshow('mask1',mask1)
#alpha=np.ones([img.shape[0], img.shape[1],1], dtype=np.uint8)
#mask=cv2.inRange(image, np.array([0,0,0]),np.array([0,0,0]),alpha);    
#cv2.bitwise_not(alpha,alpha);
#cv2.imshow('alpha',alpha)
lower_red = np.array([160,15,0])
upper_red = np.array([180,255,255])
mask2 = cv2.inRange(final,lower_red,upper_red)
cv2.imshow('mask2',mask2)

mask= mask1+mask2
cv2.imshow('total',mask)

res1 = cv2.bitwise_and(final,final, mask)
#cv2.imshow('res1', res1)
final = cv2.cvtColor(final, cv2.COLOR_HSV2BGR)
#cv2.imshow('final', final)


#print(final.shape)
#print(mask.shape)

































'''

#print(res1.shape)
res1 = cv2.bitwise_and(zero,green, None)
res2= cv2.bitwise_and(zero,blue, None)
#res3 = cv2.bitwise_xor(red,one, None)
#res3=res3*255
#res1= cv2.cvtColor(res1, cv2.COLOR_HSV2BGR)
#print(res1)
#res4 =cv2.merge((res1,res2,res3))
#print(res4)
nonzero = cv2.countNonZero(gray)
total = gray.shape[0] * gray.shape[1]
zero = total - nonzero
ratio = nonzero * 100 / float(total)
print(ratio)
'''
#cv2.imshow('h',h)
#cv2.imshow('s',s)
#cv2.imshow('v',v)
#cv2.imshow('res1',res1)
#cv2.imshow('res2',res2)
#cv2.imshow('res3',res3)
#cv2.imshow('res4',res4)
#cv2.imshow('res4',res4)
#cv2.imshow('res5',res5)
#cv2.imshow('res6',res6)
#cv2.imshow('res7',res7)
#cv2.imshow('res8',res8)
#cv2.imshow('res9',res9) 
#cv2.imshow('res10',res10)  
cv2.waitKey(0) 


cv2.destroyAllWindows()
