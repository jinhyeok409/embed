import cv2
import mediapipe as mp
import numpy as np
import requests
from datetime import datetime

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, model_complexity=2)

# 웹캠 열기
video = cv2.VideoCapture(0)

# 이전 프레임의 어깨 높이 저장 변수
previous_avg_shoulder_height = None
sudden_drop_threshold = 50  # 순간적으로 떨어진다고 판단할 픽셀 값
fall_count = 0  # 낙상 감지 횟수
fall_log_file = "fall_log.txt"  # 로그 저장 파일
blink_duration = 10  # 빨간 화면 지속 프레임 수 (약 0.3초 정도)
blink_counter = 0  # 빨간 화면 유지 타이머

def send_fall_sms():
    response = requests.post('https://textbelt.com/text', {
        'phone': '+821089398223',  # 실제 전화번호 입력 (국제번호는 불가능)
        'message': '낙상 감지됨! 긴급 확인 필요',
        'key': '',
    })
    print(response.json())

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
    landmarks = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height)))

        # 스켈레톤 그리기
        connections = mp_pose.POSE_CONNECTIONS
        for connection in connections:
            start, end = connection
            cv2.line(frame, landmarks[start], landmarks[end], (0, 255, 0), 3)

        # 어깨 좌표 추출 (왼쪽 11번, 오른쪽 12번)
        left_shoulder_y = landmarks[11][1]
        right_shoulder_y = landmarks[12][1]

        # 평균 어깨 높이 계산
        avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

        # 이전 프레임과 비교하여 순간적인 낙상 감지
        fall_detected = False
        if previous_avg_shoulder_height is not None:
            drop_amount = avg_shoulder_y - previous_avg_shoulder_height  # Y값이 커질수록 몸이 아래로 떨어짐
            if drop_amount > sudden_drop_threshold:  # 갑자기 아래로 내려간 경우
                fall_count += 1
                fall_detected = True

        # 낙상이 10번 감지되었을 때만 기록 & 카운트 초기화
        if fall_count >= 10:
            # 로그 기록
            with open(fall_log_file, "a") as file:
                file.write(f"Fall detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 화면을 빨간색으로 잠깐 변경
            blink_counter = blink_duration  
            # 문자 메시지를 보냄
            send_fall_sms()
            # 카운트 초기화
            fall_count = 0  

        # 빨간 화면 유지 시간 조절
        if blink_counter > 0:
            frame[:, :, 2] = 255  # 빨간색
            blink_counter -= 1  # 카운트 감소

        # 상태 표시
        status_text = "Fall detected!" if fall_detected else "Normal"
        color = (0, 0, 255) if fall_detected else (0, 255, 0)
        cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 현재 어깨 높이를 저장
        previous_avg_shoulder_height = avg_shoulder_y

    # 화면 출력
    cv2.imshow("Fall Detection", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

video.release()
cv2.destroyAllWindows()
