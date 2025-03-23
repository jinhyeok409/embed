import cv2
import mediapipe as mp
import time
import asyncio
from tapo import ApiClient
from tapo.requests import Color

# Tapo 계정 정보
TAPO_USERNAME = "kjh122764@naver.com"
TAPO_PASSWORD = "qq12345678"
TAPO_BULB_IP = "192.168.1.58"  # 전구의 로컬 IP 주소

# MediaPipe 손 인식 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tapo API 클라이언트 초기화
client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)
device = None  # 전구 객체를 나중에 초기화

# USB 카메라 초기화
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Cannot open camera at /dev/video0.")
    for i in range(3):  # Test indices 0, 1, 2
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap.isOpened():
            print(f"Success: Camera opened at device index {i}.")
            break
    else:
        print("Error: Could not open camera at any index.")
        exit()

# Tapo 전구 연결 (프로그램 시작 시 한 번만 실행)
async def init_tapo():
    global device
    device = await client.l530(TAPO_BULB_IP)  # L530 모델 사용

# Tapo 전구 켜기
async def turn_on_bulb():
    if device:
        await device.on()
        print("Bulb turned on!")
        await asyncio.sleep(1)

# Tapo 전구 끄기
async def turn_off_bulb():
    if device:
        await device.off()
        print("Bulb turned off!")
        await asyncio.sleep(1)

# Tapo 전구 색상 변경 (파란색)
async def set_blue_color():
    if device:
        await device.set_hue_saturation(195, 100)
        print("Bulb color set to Blue!")
        await asyncio.sleep(1)

# Tapo 전구 색상 변경 (핑크)
async def set_red_color():
    if device:
        await device.set().brightness(50).color(Color.HotPink).send(device)
        print("Bulb color set to Red!")
        await asyncio.sleep(1)

# 제스처 판별 함수
def detect_gesture(hand_landmarks):
    # 랜드마크 인덱스
    thumb_tip = hand_landmarks.landmark[4]   # 엄지 끝
    thumb_ip = hand_landmarks.landmark[3]    # 엄지 중간 관절
    index_tip = hand_landmarks.landmark[8]   # 검지 끝
    index_mcp = hand_landmarks.landmark[5]   # 검지 기저부
    middle_tip = hand_landmarks.landmark[12] # 중지 끝
    middle_mcp = hand_landmarks.landmark[9]  # 중지 기저부
    ring_tip = hand_landmarks.landmark[16]   # 약지 끝
    ring_mcp = hand_landmarks.landmark[13]   # 약지 기저부
    pinky_tip = hand_landmarks.landmark[20]  # 새끼 끝
    pinky_mcp = hand_landmarks.landmark[17]  # 새끼 기저부

    # Thumb Up (불 켜기): 엄지가 가장 위에 있고 나머지 손가락이 굽어 있음
    thumb_up = (
        thumb_tip.y < index_tip.y and
        thumb_tip.y < middle_tip.y and
        thumb_tip.y < ring_tip.y and
        thumb_tip.y < pinky_tip.y and
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )

    # Thumb Down (불 끄기): 엄지가 가장 아래에 있고 나머지 손가락이 굽어 있음
    thumb_down = (
        thumb_tip.y > index_tip.y and
        thumb_tip.y > middle_tip.y and
        thumb_tip.y > ring_tip.y and
        thumb_tip.y > pinky_tip.y and
        index_tip.y > index_mcp.y and
        middle_tip.y > middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y
    )

    # Index Up (파란색): 검지만 펴져 있고 나머지 손가락은 굽어 있음
    index_up = (
        index_tip.y < index_mcp.y and  # 검지가 펴져 있음
        middle_tip.y > middle_mcp.y and  # 중지는 굽어 있음
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y and
        thumb_tip.y > thumb_ip.y  # 엄지가 아래로
    )

    # V Sign (빨간색): 검지와 중지가 펴져 있고 나머지 손가락은 굽어 있음
    v_sign = (
        index_tip.y < index_mcp.y and
        middle_tip.y < middle_mcp.y and
        ring_tip.y > ring_mcp.y and
        pinky_tip.y > pinky_mcp.y and
        thumb_tip.y > thumb_ip.y
    )

    if thumb_up:
        return "Thumb Up"
    elif thumb_down:
        return "Thumb Down"
    elif index_up:
        return "Index Up"
    elif v_sign:
        return "V Sign"
    return ""

# Tapo 초기화 실행
asyncio.run(init_tapo())

# 메인 루프
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame.")
        break

    # BGR -> RGB 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 인식 수행
    results = hands.process(rgb_frame)

    # 결과 그리기 및 제스처 판별
    gesture_text = ""
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture_text = detect_gesture(hand_landmarks)

            # 제스처에 따라 전구 제어
            if gesture_text == "Thumb Up":
                asyncio.run(turn_on_bulb())  # 엄지 척 -> 전구 켜기
            elif gesture_text == "Thumb Down":
                asyncio.run(turn_off_bulb())  # 엄지 아래 -> 전구 끄기
            elif gesture_text == "Index Up":
                asyncio.run(set_blue_color())  # 검지만 위 -> 파란색
            elif gesture_text == "V Sign":
                asyncio.run(set_red_color())  # 검지와 중지 위 -> 빨간색

    # FPS 표시
    xpos = 50
    ypos = 50
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (xpos, ypos), font, fontScale, color, thickness)

    # 제스처 텍스트 표시
    if gesture_text:
        cv2.putText(frame, gesture_text, (xpos, ypos + 40), font, fontScale, (0, 0, 255), thickness)

    # 화면에 출력
    cv2.imshow("Hand Tracking", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
