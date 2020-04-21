import cv2
import numpy as np
cropping = False
ix = -1
iy = -1
refPt = []
img = cv2.imread("sample.png")
copy = img.copy()

def crop(event,x,y,flags,param):
    global ix,iy,cropping,refPt
    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        ix,iy = x,y
        refPt=[(x,y)]
    elif event == cv2.EVENT_LBUTTONUP:
         cropping = False
         copy1 = img
         cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),thickness=2)
         refPt.append((x,y))
cv2.namedWindow("img")
cv2.setMouseCallback('img',crop)
while True:
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
        # reset
        if key == ord("r"):
            img = copy.copy()
        # crop
        elif key == ord("c"):
            break
if len(refPt) == 2:
    img2 = copy[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    cv2.imshow("cropped", img2)
    cv2.imwrite("cropped.png", img2)
    cv2.waitKey(0)






cv2.destroyAllWindows()

