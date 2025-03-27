# USB 카메라 초기화
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Cannot open camera at /dev/video0.")
    for i in range(3):  # Test indices 0, 1, 2
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if cap.isOpened():
            print(f"Success: Camera opened at device index {i} with resolution 320x240.")
            break
    else:
        print("Error: Could not open camera at any index.")
        exit()

# 설정된 해상도 출력
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Current resolution: {width}x{height}")
