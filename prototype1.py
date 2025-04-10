import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from datetime import datetime

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=0)

# Open the camera (device 0, using V4L2 backend)
video = cv2.VideoCapture(0, cv2.CAP_V4L2)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
video.set(cv2.CAP_PROP_FPS, 30)

# Print actual camera settings
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("Actual FPS:", fps)
print("Actual Width:", width)
print("Actual Height:", height)

# Setup VideoWriter to save output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Can also use 'MJPG', 'mp4v'
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# Fall detection variables
previous_avg_shoulder_height = None
sudden_drop_threshold = 30
fall_count = 0
fall_log_file = "fall_log.txt"
blink_duration = 10
blink_counter = 0

# Function to send SMS when a fall is detected
def send_fall_sms():
    response = requests.post('https://textbelt.com/text', {
        'phone': '+821096900339',  # Replace with actual phone number
        'message': 'Fall detected! Please check immediately.',
        'key': '',  # Your Textbelt API key
    })
    print(response.json())

# FPS calculation
prev_time = time.time()
frame_skip = 2
frame_count = 0

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip this frame to improve performance

    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get shoulder positions
        left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
        right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))

        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

        fall_detected = False
        if previous_avg_shoulder_height is not None:
            drop_amount = avg_shoulder_y - previous_avg_shoulder_height
            if drop_amount > sudden_drop_threshold:
                fall_count += 1
                fall_detected = True

        # If fall condition is met, log and send SMS
        if fall_count >= 10:
            with open(fall_log_file, "a") as file:
                file.write(f"Fall detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            send_fall_sms()  # Send alert
            blink_counter = blink_duration  # Start blinking effect
            fall_count = 0  # Reset counter

        # Draw a line between shoulders
        cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 3)

        # Blink effect (red overlay)
        if blink_counter > 0:
            frame[:, :, 2] = 255
            blink_counter -= 1

        # Display status on screen
        status_text = "Fall detected!" if fall_detected else "Normal"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        previous_avg_shoulder_height = avg_shoulder_y

    # Calculate FPS
    curr_time = time.time()
    current_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_text = f"FPS: {int(current_fps)}"
    cv2.putText(frame, fps_text, (width - 150, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the output frame
    cv2.imshow("Fall Detection", frame)

    # Save the frame to the output video
    out.write(frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
video.release()
out.release()
cv2.destroyAllWindows()
