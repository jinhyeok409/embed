import cv2
import mediapipe as mp
import numpy as np
import time
import requests
from datetime import datetime

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)

# 웹캠 또는 영상 열기
video = cv2.VideoCapture("0")
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

# 실제 설정된 값 확인
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

print("Actual FPS:", fps)
print("Actual Width:", width)
print("Actual Height:", height)

# 영상 저장을 위한 VideoWriter 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 또는 'MJPG', 'mp4v'
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

# 낙상 관련 변수
previous_avg_shoulder_height = None
sudden_drop_threshold = 30
fall_count = 0
fall_log_file = "fall_log.txt"
blink_duration = 10
blink_counter = 0

# SMS 전송 함수
def send_fall_sms():
    response = requests.post('https://textbelt.com/text', {
        'phone': '+821096900339',  # 실제 전화번호로 교체 필요
        'message': '낙상 감지됨! 긴급 확인 필요',
        'key': '',
    })
    print(response.json())

# FPS 측정용 시간
prev_time = time.time()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_shoulder = (int(landmarks[11].x * width), int(landmarks[11].y * height))
        right_shoulder = (int(landmarks[12].x * width), int(landmarks[12].y * height))

        avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

        fall_detected = False
        if previous_avg_shoulder_height is not None:
            drop_amount = avg_shoulder_y - previous_avg_shoulder_height
            if drop_amount > sudden_drop_threshold:
                fall_count += 1
                fall_detected = True

        # 낙상 조건 만족 시 SMS 전송 및 로그 기록
        if fall_count >= 10:
            with open(fall_log_file, "a") as file:
                file.write(f"Fall detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # SMS 전송
            send_fall_sms()

            # 화면 깜빡임 효과
            blink_counter = blink_duration
            fall_count = 0

        # 어깨 선만 그리기
        cv2.line(frame, left_shoulder, right_shoulder, (255, 0, 0), 3)

        if blink_counter > 0:
            frame[:, :, 2] = 255
            blink_counter -= 1

        status_text = "Fall detected!" if fall_detected else "Normal"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        previous_avg_shoulder_height = avg_shoulder_y

    # FPS 계산 및 출력
    curr_time = time.time()
    current_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_text = f"FPS: {int(current_fps)}"
    cv2.putText(frame, fps_text, (width - 150, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 출력 화면 보여주기
    cv2.imshow("Fall Detection", frame)

    # 프레임 저장
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# 종료 처리
video.release()
out.release()  # 영상 저장 마무리
cv2.destroyAllWindows()
