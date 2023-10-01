import cv2 as cv
import numpy as np
import os
import math
from frame_operations import FrameOperations

class PoseEstimator():

    def __init__(self):
        self.FRAME_OPS = FrameOperations()

        self.keypoints: {
                        0: "nose",
                        1: "left_eye",
                        2: "right_eye",
                        3: "left_ear",
                        4: "right_ear",
                        5: "left_shoulder",
                        6: "right_shoulder",
                        7: "left_elbow",
                        8: "right_elbow",
                        9: "left_wrist",
                        10: "right_wrist",
                        11: "left_hip",
                        12: "right_hip",
                        13: "left_knee",
                        14: "right_knee",
                        15: "left_ankle",
                        16: "right_ankle"
            }
             self.POSE_PAIRS = [ ["left_shoulder", "right_shoulder"], ["left_hip", "right_hip"], ["left_knee", "right_knee"]]
              self.CWD = os.getcwd()
        self.RESOURCES = os.path.join(self.CWD,'resources')
        self.GRAPH_OPT = os.path.join(self.RESOURCES,'graph_opt.pb')

        self.NET = cv.dnn.readNetFromTensorflow(self.GRAPH_OPT)
        self.THR = 0.1
        self.IN_WIDTH = 396
        self.IN_HEIGHT = 368

        self.POINTS = []
         self.KEY_DISTANCES = {"Shoulders":{"left_shoulder-right_shoulder":None},
        "Hip":{"left_hip-right_hip":None},
        "Knee":{"left_knee-right_knee":None}}

        self.KEY_ANGLES = {"Shoulders": [],"Hip":[],"Knee":[]}

        self.TEXT_COLOR = (0,0,0)

    def rad_to_deg(self,rad):
        return rad * (180/math.pi)

    def get_pose_key_angles(self, frame, wantBlank = False):
        """applies pose estimation on frame, gets the distances between points"""

        # for the key points that do not come in pairs
        left_eye_pos = None
        right_eye_pos = None

        left_ear = None
        right_ear = None
        
        
        left_elbow = None
        right_elbow = None


        left_wrist=None
        right_wrist=None

        left_ankle=None
        right_ankle=None


        frame_h,frame_w = frame.shape[0:2]

         self.NET.setInput(cv.dnn.blobFromImage(frame, 1.0, (self.IN_WIDTH, self.IN_HEIGHT), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = self.NET.forward()

        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(self.BODY_PARTS) == out.shape[1])

        #clear to get new points
        self.POINTS.clear()

        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frame_w * point[0]) / out.shape[3]
            y = (frame_h * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            if(conf > self.THR):
                self.POINTS.append((int(x),int(y)))
            else:
                self.POINTS.append(None)

        # create blank frame overlay once OpenPose has read original frame so as to work
        if wantBlank:

            frame = np.zeros((frame_h,frame_w,3),np.uint8)

            self.TEXT_COLOR = (255,255,255)

        for pair in self.POSE_PAIRS:
            # ex: pair 1: [["Neck","RShoulder"]]
            # partFrom = Neck, partTo = RShoulder
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in self.BODY_PARTS)
            assert(partTo in self.BODY_PARTS)


            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            # if found points (if not found, returns None)
            if self.POINTS[idFrom] and self.POINTS[idTo]:
                

                # now we check each of the key points.
                # "a", "b" correspond to the lengths of the limbs, "c" is the length between the end dots on the triangle. See video.
                # we use law of cosines to find angle c: 
                # cos(C) = (a^2 + b^2 - c^2) / 2ab
                # we first check for the points that do not come in pairs (make up the longest side of the triangle in the vid)
            
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            # if found points (if not found, returns None)
            if self.POINTS[idFrom] and self.POINTS[idTo]:
                

                # now we check each of the key points.
                # "a", "b" correspond to the lengths of the limbs, "c" is the length between the end dots on the triangle. See video.
                # we use law of cosines to find angle c: 
                # cos(C) = (a^2 + b^2 - c^2) / 2ab
                # we first check for the points that do not come in pairs (make up the longest side of the triangle in the vid)