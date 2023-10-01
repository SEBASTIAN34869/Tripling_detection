import os
import sys
import ast
import cv2
import torch
import numpy as np
from AI.tripling_detection.SimpleHRNet import SimpleHRNet
from statistics import mode,mean
import time

keypoints={
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


def calculate_angle(p1,p2,p3):

    a=np.array(p1)
    b=np.array(p2)
    c=np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    angle = np.arccos(cosine_angle) 
    angle_new=np.degrees(angle)
    
    return angle_new


def pose_track(filename):
    start = time.time()
    # print("inside pose track")
    device=None
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    image_resolution = ast.literal_eval('(384, 288)')
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    
    ### Tiny yolo-v3 is faster.
    # yolo_model_def="./models/detectors/yolo/config/yolov3-tiny.cfg"
    yolo_class_path="./AI/tripling_detection/models/detectors/yolo/data/coco.names"
    # yolo_weights_path="./models/detectors/yolo/weights/yolov3-tiny.weights"
    yolo_model_def="./AI/tripling_detection/models/detectors/yolo/config/yolov3.cfg"
    yolo_weights_path="./AI/tripling_detection/models/detectors/yolo/weights/yolov3.weights"
    
    model_start = time.time()     
    model = SimpleHRNet(
        c=48,
        nof_joints=17,
        checkpoint_path='./AI/tripling_detection/weights/pose_hrnet_w48_384x288.pth',
        model_name='HRNet',
        resolution=image_resolution,
        multiperson=True,
        return_bounding_boxes=False,
        max_batch_size=16,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
    )
    model_end = time.time()
    print(f"Model loding time Pose estimation : {model_end - model_start}")
        
    if filename is not None:
        frame=cv2.imread(filename)
    else:
        print("Incorrect file format")
    
    pts = model.predict(frame)
    person_ids = np.arange(len(pts), dtype=np.int32)

    lknee_sitting = []
    rknee_sitting = []
    lhip_sitting = []
    rhip_sitting = []
    knee_threshold = 140
    hip_threshold = 130

    for i, (pt, pid) in enumerate(zip(pts, person_ids)):
    # for (pt, person_ids) in enumerate(zip(pts, person_ids)):
        # frame = draw_points_and_skeleton(frame, pt, joints_dict()['coco']['skeleton'], person_index=pid,
        #                         points_color_palette='gist_rainbow', skeleton_color_palette='jet',
        #                         points_palette_samples=10)
        
        lknee = calculate_angle(pt[11],pt[13],pt[15])
        rknee = calculate_angle(pt[12],pt[14],pt[16])
        rhip = calculate_angle(pt[6],pt[12],pt[14])
        lhip = calculate_angle(pt[5],pt[11],pt[13])
         
        if lknee > knee_threshold and rknee > knee_threshold:
            pass
        else:
            lknee_sitting.append(lknee)
            rknee_sitting.append(rknee)
        
        if lhip > hip_threshold and rhip > hip_threshold:
            pass  
        else:
            lhip_sitting.append(lhip)
            rhip_sitting.append(rhip)

    temp_final_list=[]

    temp_final_list.append(len(lknee_sitting))
    temp_final_list.append(len(rknee_sitting))
    temp_final_list.append(len(lhip_sitting))
    temp_final_list.append(len(rhip_sitting))


    try:
        num_of_people = mode(temp_final_list)
    except:
        avg = mean(temp_final_list)
        num_of_people = int(avg)

 
    end=time.time()  
    print(f"Time taken for pose estimation :{time.time() - start}")
    return num_of_people





