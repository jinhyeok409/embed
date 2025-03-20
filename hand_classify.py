import cv2
import mediapipe as mp
import numpy as np
import math

# 손 인식 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 손가락 각도 계산 함수
def calculate_angle(a, b, c):
    """
    손가락 마디의 각도를 계산하는 함수.
    a, b, c는 landmark 좌표 (x, y) 튜플.
    """
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # acos의 범위를 제한하여 에러 방지
    return np.degrees(angle)  # 라디안 -> 도(degree) 변환

# 손동작 분류 함수
def classify_hand_gesture(landmarks):
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # 손가락 마디 좌표
    index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    # 손가락 각도 계산
    index_angle = calculate_angle(index_mcp, landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP], landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP])
    middle_angle = calculate_angle(middle_mcp, landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP], landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP])
    ring_angle = calculate_angle(ring_mcp, landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP], landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP])
    pinky_angle = calculate_angle(pinky_mcp, landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP], landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP])

    # 엄지 각도 계산
    thumb_angle = calculate_angle(thumb_mcp, thumb_ip, thumb_tip)

    # 주먹(Fist) 판별
    is_fist = all(angle < 50 for angle in [index_angle, middle_angle, ring_angle, pinky_angle])

    # V Sign 판별
    is_v = index_angle > 60 and middle_angle > 60 and math.dist([landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y],
                                                                 [landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y]) > 0.05

    # 손바닥 펼침(Open Hand)
    is_open = all(angle > 160 for angle in [index_angle, middle_angle, ring_angle, pinky_angle])

    # 엄지척(Thumb Up) 개선
    is_thumb_up = (thumb_tip.y < wrist.y and thumb_angle > 120 and 
                   all(angle < 70 for angle in [index_angle, middle_angle, ring_angle, pinky_angle]))

    if is_fist:
        return "Fist"
    elif is_v:
        return "V Sign"
    elif is_open:
        return "Open Hand"
    elif is_thumb_up:
        return "Thumb Up"
    else:
        return "Unknown"


# 웹캠 열기
cap = cv2.VideoCapture(0)

# 손 인식 모델 초기화
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("웹캠을 열 수 없습니다.")
            break

        # 이미지 전처리 (RGB 변환)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 손 인식 실행
        results = hands.process(frame_rgb)

        # 손이 인식되면 손동작 분류
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 손동작 분류
                gesture = classify_hand_gesture(landmarks)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 실시간 프레임 표시
        cv2.imshow("Hand Gesture Recognition", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 웹캠 종료
cap.release()
cv2.destroyAllWindows()
