import cv2
import os
import numpy as np
import mediapipe as mp
from datetime import datetime

# -------- 설정 --------
SAVE_DIR = "dataset"
LABEL = "fall"  # ← 'fall' 또는 'normal' 로 수집 전 변경
SEQ_LENGTH = 90  # 시퀀스 길이 (프레임 수, 약 3초)
CAM_ID = 0  # 기본 웹캠 번호
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# -------- 저장 디렉토리 준비 --------
os.makedirs(os.path.join(SAVE_DIR, LABEL), exist_ok=True)

# -------- MediaPipe 초기화 --------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------- 웹캠 준비 --------
cap = cv2.VideoCapture(CAM_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

print("[INFO] 웹캠 시작됨. 's' 키를 누르면 수집 / 'q'로 종료")

while True:
    frames = []
    count = 0

    # ----- 사용자 입력 대기 -----
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] 카메라에서 프레임을 읽을 수 없습니다.")
            break

        cv2.putText(frame, f"Label: {LABEL} | Press 's' to record {SEQ_LENGTH} frames", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Pose Capture", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            print(f"[INFO] '{LABEL}' 데이터 수집 시작")
            break
        elif key == ord('q'):
            print("[INFO] 종료 중...")
            cap.release()
            cv2.destroyAllWindows()
            pose.close()
            exit()

    # ----- 시퀀스 수집 시작 -----
    while count < SEQ_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(image_rgb)

        if result.pose_landmarks:
            keypoints = result.pose_landmarks.landmark
            coords = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in keypoints])
            frames.append(coords.flatten())
            count += 1

            mp.solutions.drawing_utils.draw_landmarks(
                frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        else:
            cv2.putText(frame, "Pose Not Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Frames Collected: {count}/{SEQ_LENGTH}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Pose Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ----- 저장 -----
    if len(frames) == SEQ_LENGTH:
        frames_np = np.stack(frames)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, LABEL, f"{LABEL}_{timestamp}.npy")
        np.save(filename, frames_np)
        print(f"[SAVED] {filename}")
    else:
        print("[INFO] 시퀀스 수집 실패 (감지된 프레임 부족)")

