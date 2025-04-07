import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=1)

# 웹캠 열기 및 설정
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
video.set(cv2.CAP_PROP_FPS, 30)

# 실제 설정된 값 확인
print("Actual FPS:", video.get(cv2.CAP_PROP_FPS))
print("Actual Width:", video.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Actual Height:", video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 이전 프레임의 어깨 높이 저장 변수
previous_avg_shoulder_height = None
sudden_drop_threshold = 50  # 순간적으로 떨어진다고 판단할 픽셀 값
fall_count = 0  # 낙상 감지 횟수
fall_log_file = "fall_log.txt"  # 로그 저장 파일
blink_duration = 10  # 빨간 화면 지속 프레임 수 (약 0.3초 정도)
blink_counter = 0  # 빨간 화면 유지 타이머

# FPS 측정을 위한 시간 변수 초기화
prev_time = time.time()

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    # 좌우 반전
    frame = cv2.flip(frame, 1)

    # 포즈 감지
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    height, width, _ = frame.shape

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 어깨 좌표만 추출 (왼쪽 11번, 오른쪽 12번)
        left_shoulder_y = int(landmarks[11].y * height)
        right_shoulder_y = int(landmarks[12].y * height)

        # 평균 어깨 높이 계산
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        # 이전 프레임과 비교하여 순간적인 낙상 감지
        fall_detected = False
        if previous_avg_shoulder_height is not None:
            drop_amount = avg_shoulder_y - previous_avg_shoulder_height
            if drop_amount > sudden_drop_threshold:
                fall_count += 1
                fall_detected = True

        # 낙상이 10번 감지되었을 때만 기록 & 카운트 초기화
        if fall_count >= 10:
            with open(fall_log_file, "a") as file:
                file.write(f"Fall detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            blink_counter = blink_duration
            fall_count = 0

        # 빨간 화면 유지 시간 조절
        if blink_counter > 0:
            frame[:, :, 2] = 255  # 빨간색
            blink_counter -= 1

        # 상태 표시
        status_text = "Fall detected!" if fall_detected else "Normal"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 현재 어깨 높이 저장
        previous_avg_shoulder_height = avg_shoulder_y

    # FPS 계산 및 출력
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_text, (width - 150, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 화면 출력
    cv2.imshow("Fall Detection", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
