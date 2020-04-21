import cv2
import numpy as np
drawing = False
mode = True
ix = -1
iy = -1
pts = []
img = cv2.imread("sample.png")
copy = img.copy()



def draw(event,x,y,flags,param):
        global ix,iy,drawing, mode,pts
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            ix,iy=x,y
            pts.append([ix,iy])

        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                cv2.line(img,(ix,iy),(x,y),(0,0,255),5)
                ix,iy = x,y
                pts.append([ix, iy])
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            cv2.line(img,(ix,iy),(x,y),(0,0,255),5)
            ix, iy = x, y
            pts.append([ix, iy])

        return x,y


cv2.namedWindow("image")
cv2.setMouseCallback('image',draw)
while(1):
    cv2.imshow("image", img)
    key = cv2.waitKey(1) & 0xFF
    # reset
    if key == ord("r"):
        img = copy.copy()
    # crop
    elif key == ord("c"):
        break

pts = np.array(pts)
print(pts)
cv2.fillPoly(img, [pts], (0, 0, 255))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 120, 70])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

mask1 = mask1 + mask2

res1 = cv2.bitwise_and(copy, copy, mask=mask1)


cv2.imshow("er", res1)
cv2.imwrite("cropped.png", res1)
cv2.waitKey(0)
cv2.destroyAllWindows()