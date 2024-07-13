import cv2
import mediapipe as mp
import holosense_libconfigs
from holosense import SpatialTracker
import numpy as np
import math
import matplotlib.pyplot as plt



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
spatial_tracker = SpatialTracker(
    fov=78.5,
    aspectratio=16/9,
    eyedistance=4,
    eyenosedistance=3,
    single_output=False)


LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
FIXED_POINTS_LIST = []
print("setup complete")



frame = cv2.imread("frontal_face.png")

frame = cv2.flip(frame, 1)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img_h, img_w = frame.shape[:2]
results = face_mesh.process(rgb_frame)
prev_points=np.array([[p.x, p.y, p.z, 1] for p in results.multi_face_landmarks[0].landmark])
previous_average = np.sum(prev_points, axis=0)/len(prev_points)
lefteye_location_calibration = np.sum(prev_points[LEFT_IRIS], axis=0)/4
righteye_location_calibration = np.sum(prev_points[RIGHT_IRIS], axis=0)/4


cap = cv2.VideoCapture("/dev/video0")

succ = True
while succ:
    succ, frame = cap.read()
    if not succ:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(rgb_frame)
    if not results.multi_face_landmarks:
        continue
    mesh_points=np.array([[p.x, p.y, p.z, 1] for p in results.multi_face_landmarks[0].landmark])
    Transform = np.linalg.pinv(np.linalg.pinv(prev_points).dot(mesh_points))
    transformed = []
    for i in mesh_points:
        i.shape = (1,4)
        transformed.append((i.dot(Transform)).reshape(4,))
    averageY = np.sum(transformed, axis=0)/len(transformed)
    (right_pos, left_pos, N_pos) = spatial_tracker.calculatePosition(results.multi_face_landmarks[0])

    

    # approximate positions of centres of the eyeballs
    OS_pos = (right_pos + left_pos*3)/4
    OD_pos = (right_pos*3 + left_pos)/4
    
    rlv = (left_pos - right_pos)
    rnv = N_pos-right_pos
    p_vec = np.cross(rnv, rlv)
    p_vec /= np.linalg.norm(p_vec)
    plane_origin = right_pos + p_vec






    lefteye_location = np.sum(np.array(transformed)[LEFT_IRIS], axis=0)/4
    righteye_location = np.sum(np.array(transformed)[RIGHT_IRIS], axis=0)/4    
    print("Left eye coordinate:" + str(lefteye_location))
    print("Right eye coordinate:" + str(righteye_location))
    #print(np.sum(mesh_points, axis=0)/len(transformed))

    


