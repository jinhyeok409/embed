import cv2
import numpy as np
import mediapipe as mp
from time import time

mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=2)

previous_avg_shoulder_height = 0
def detectPose(frame, pose_model):
    frame = cv2.flip(frame, 1)  # 화면 좌우 반전
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)
    height, width, _ = frame.shape
    landmarks = []
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height)))
        
        # 스켈레톤 표시
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return None, frame
    return landmarks, frame

def detectFall(landmarks, previous_avg_shoulder_height):
    left_shoulder_y = landmarks[11][1]
    right_shoulder_y = landmarks[12][1]
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
    
    if previous_avg_shoulder_height == 0:
        previous_avg_shoulder_height = avg_shoulder_y
        return False, previous_avg_shoulder_height
    
    fall_threshold = previous_avg_shoulder_height * 1.5
    sudden_drop_threshold = previous_avg_shoulder_height * 1.2  # 순간적으로 낮아지는 경우 감지
    
    if avg_shoulder_y > fall_threshold and avg_shoulder_y - previous_avg_shoulder_height > sudden_drop_threshold:
        return True, avg_shoulder_y
    return False, avg_shoulder_y

time1 = 0
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break
    
    landmarks, frame = detectPose(frame, pose_video)
    time2 = time()
    
    if (time2 - time1) > 2 and landmarks is not None:
        fall_detected, previous_avg_shoulder_height = detectFall(landmarks, previous_avg_shoulder_height)
        status_text = "Fall detected!" if fall_detected else "Normal"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        if fall_detected:
            print("Fall detected!")
        time1 = time2
    
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
