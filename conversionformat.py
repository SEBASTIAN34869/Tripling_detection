import cv2 
image=cv2.imread('D:/project/simple-HRNet/phase3test1.jpg')
image_bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
cv2.imwrite('converted.png',image_bgr)
