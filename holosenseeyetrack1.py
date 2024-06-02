from operator import rshift
import cv2 as cv 
import numpy as np
import mediapipe as mp 
import math
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE= [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246] 
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(np.float32) for p in results.multi_face_landmarks[0].landmark])
            # fit a circle to the cornea from the picture
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            # extract some coordinates for processing
            lec = list(mesh_points[263])
            lec1 = list(mesh_points[398])
            rec = list(mesh_points[33])
            rec1 = list(mesh_points[173])
            nc = list(mesh_points[18])
            nc1 = list(mesh_points[8])

            # center points of each eye 
            lec_c = [(lec[0] + lec1[0])/2, (lec[1] + lec1[1])/2]
            rec_c = [(rec[0] + rec1[0])/2, (rec[1] + rec1[1])/2]
            angle = math.atan((lec[1]-lec1[1])/(lec[0]-lec1[0]))
            
            # find the matrix for rotation to line it up
            angle += 2 * 3.14159
            if angle >= 2*3.14159:
                angle -= 2 * 3.14159
            deg = (angle * (180/3.14159))
            M = cv.getRotationMatrix2D((lec_c[0] - min(lec1[0], lec[0]) , lec_c[1] - (min(lec[1], lec1[1]) -50)), deg, 1.0)
            # circumference coordinate that is along the line
            
            h_cc = [l_cx, l_cy]

            t_cc = [M[0][0] * h_cc[0] + M[0][1] * h_cc[1] + M[0][2], M[1][0] * h_cc[1] + M[1][1] * h_cc[1] + M[1][2]] 
            t_lec = [M[0][0] * lec[0] + M[0][1] * lec[1] + M[0][2], M[1][0] * lec[1] + M[1][1] * lec[1] + M[1][2]]            
            t_lec1 = [M[0][0] * lec1[0] + M[0][1] * lec1[1] + M[0][2], M[1][0] * lec1[1] + M[1][1] * lec1[1] + M[1][2]]


            print(t_cc[0] - lec_c[0])
            #print(t_cc[1] - lec_c[1])

            angle = math.atan((rec[1]-rec1[1])/(rec[0]-rec1[0]))
            deg = (angle * (180/3.14159))
            if deg >= 360:
                deg -= 360


            M1 = cv.getRotationMatrix2D((rec_c[0] - min(rec1[0], rec[0]) , rec_c[1] - (min(rec[1], rec1[1]) -50)), deg, 1.0)


            cc = [lec[0] + l_radius * math.cos(angle), lec[1] + l_radius * math.sin(angle)]
            
            t_cc = [M1[0][0] * cc[0] + M[0][1] * cc[1] + M1[0][2], M1[1][0] * cc[1] + M1[1][1] * cc[1] + M1[1][2]]
            t_rec = [M1[0][0] * rec[0] + M1[0][1] * rec[1] + M1[0][2], M1[1][0] * rec[1] + M1[1][1] * rec[1] + M1[1][2]]
            t_rec1 = [M1[0][0] * rec1[0] + M1[0][1] * rec1[1] + M1[0][2], M1[1][0] * rec1[1] + M1[1][1] * rec1[1] + M1[1][2]]
            

            # the matrix transformation will rotate the coordinate system so the center of the line is (0,0) and that line is parallel to the X axis.
            # it will be scaled so the eye itself is a fixed 100 px wide, and the height is 20px.
            # affine warping will be done for visualization but can be removed for tracking with higher FPS
            
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            #cv.line(frame, [int(i) for i in lec], [int(i) for i in lec1], (255,0,0), 2, cv.LINE_AA)
            #cv.line(frame, [int(i) for i in rec], [int(i) for i in rec1], (255,0,0), 2, cv.LINE_AA)

            #cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA)
            #cv.circle(frame, center_left, 3, (0,0,255), -1)
            #cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA)
            #cv.circle(frame, center_right, 3, (0,0,255), -1)
            try:
                i1 = frame[int(min(lec[1], lec1[1])) -50 :int(max(lec[1], lec1[1]))+50, int(min(lec1[0], lec[0])):int(max(lec[0], lec1[0]))]            
                newimage = cv.warpAffine(i1, M, (int(abs(t_lec[0]-t_lec1[0])), 120))
                cv.imshow("Left eye", newimage)
                i2 = frame[int(min(rec[1], rec1[1])) -50 :int(max(rec[1], rec1[1]))+50, int(min(rec1[0], rec[0])):int(max(rec[0], rec1[0]))]            
                newimage1 = cv.warpAffine(i2, M1, (int(abs(t_rec[0]-t_rec1[0])), 120))
                cv.imshow("Right eye", newimage1)
            except:
                print("oh no")                

        cv.imshow('eye tracking system', frame)

        
        key = cv.waitKey(1)
        if key ==ord('q'):
            cv.imwrite("lefteye.png", newimage)
            cv.imwrite("righteye.png", newimage1)
            break
cap.release()
cv.destroyAllWindows()