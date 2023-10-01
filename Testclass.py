import cv2
from SimpleHRNet import SimpleHRNet

model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth")
image = cv2.imread("D:/Testing Images/phase3test1.jpg", cv2.IMREAD_COLOR)

joints = model.predict(image)