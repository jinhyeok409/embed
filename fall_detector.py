# fall_detector.py

import argparse
import collections
from functools import partial
import time
import svgwrite
import cv2
from datetime import datetime
import os
import threading
import queue

# gstreamer.py와 pose_engine.py가 같은 폴더에 있다고 가정
import gstreamer
from pose_engine import PoseEngine
from pose_engine import KeypointType

EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_SHOULDER),
    (KeypointType.NOSE, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
)

def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))

def draw_pose(dwg, pose, src_size, inference_box, color='yellow', threshold=0.2):
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold:
            continue
        kp_x = int((keypoint.point[0] - box_x) * scale_x)
        kp_y = int((keypoint.point[1] - box_y) * scale_y)
        xys[label] = (kp_x, kp_y)
        dwg.add(dwg.circle(center=(kp_x, kp_y), r=5,
                           fill='none', stroke='none', display='none'))

    for a, b in EDGES:
        if a not in xys or b not in xys:
            continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0
    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def run(inf_callback, render_callback):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    gstreamer.run_pipeline(partial(inf_callback, engine),
                           partial(render_callback, engine),
                           src_size, inference_size,
                           mirror=args.mirror,
                           videosrc=args.videosrc,
                           h264=args.h264,
                           jpeg=args.jpeg)

def main():
    # --- 상태 변수 및 설정 ---
    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    fps_counter = avg_fps_counter(30)

    # --- 넘어짐 감지 관련 변수 ---
    shoulder_y_history = collections.deque(maxlen=10) # 어깨 Y좌표 10프레임 저장
    FALL_THRESHOLD = 50  # Y 좌표 변화량 기준 (환경에 맞게 조절 필요)
    fall_detected_time = 0 # 마지막으로 넘어짐이 감지된 시간
    FALL_COOLDOWN_SECONDS = 5.0 # 5초에 한번만 저장

    # --- 이미지 저장을 위한 스레드 및 큐 설정 ---
    save_queue = queue.Queue()

    def image_save_worker():
        if not os.path.exists("fall_images"):
            os.makedirs("fall_images")
        while True:
            try:
                frame_to_save = save_queue.get(timeout=1)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join("fall_images", f"fall_{timestamp}.jpg")
                cv2.imwrite(filename, cv2.cvtColor(frame_to_save, cv2.COLOR_RGB2BGR))
                print(f"이미지 저장 완료: {filename}")
                save_queue.task_done()
            except queue.Empty:
                continue
    
    # 데몬 스레드로 이미지 저장 워커 시작
    threading.Thread(target=image_save_worker, daemon=True).start()


    # --- GStreamer 콜백 함수 정의 ---
    # fall_detector.py
    def run_inference(engine, input_tensor):
        # 3D numpy 배열을 1D로 펼쳐서 전달
        return engine.run_inference(input_tensor.flatten())
    def render_overlay(engine, output, src_size, inference_box, frame):
        nonlocal n, sum_process_time, sum_inference_time, fps_counter
        nonlocal shoulder_y_history, fall_detected_time

        svg_canvas = svgwrite.Drawing('', size=src_size)
        start_time = time.monotonic()
        outputs, inference_time = engine.ParseOutput()
        end_time = time.monotonic()

        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time * 1000
        avg_inference_time = sum_inference_time / n if n > 0 else 0

        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time if avg_inference_time > 0 else 0,
            next(fps_counter), len(outputs)
        )
        shadow_text(svg_canvas, 10, 20, text_line)

        fall_detected_in_frame = False
        for pose in outputs:
            draw_pose(svg_canvas, pose, src_size, inference_box)

            ls = pose.keypoints.get(KeypointType.LEFT_SHOULDER)
            rs = pose.keypoints.get(KeypointType.RIGHT_SHOULDER)

            if ls and rs and ls.score > 0.5 and rs.score > 0.5:
                # inference_box를 기준으로 좌표 스케일링
                box_x, box_y, box_w, box_h = inference_box
                scale_y = src_size[1] / box_h
                
                ls_y = (ls.point[1] - box_y) * scale_y
                rs_y = (rs.point[1] - box_y) * scale_y
                shoulder_y = (ls_y + rs_y) / 2
                shoulder_y_history.append(shoulder_y)

                if len(shoulder_y_history) == shoulder_y_history.maxlen:
                    delta = shoulder_y_history[-1] - shoulder_y_history[0]
                    if delta > FALL_THRESHOLD:
                        fall_detected_in_frame = True
        
        current_time = time.monotonic()
        if fall_detected_in_frame and (current_time - fall_detected_time > FALL_COOLDOWN_SECONDS):
            fall_detected_time = current_time # 감지 시간 업데이트
            shadow_text(svg_canvas, 10, 50, "넘어짐 감지!", font_size=24)
            # 큐에 저장할 프레임 추가 (복사본을 전달)
            if not save_queue.full():
                save_queue.put(frame.copy())
        
        return (svg_canvas.tostring(), False)

    try:
        run(run_inference, render_overlay)
    except KeyboardInterrupt:
        print("\n프로그램 종료.")

if __name__ == '__main__':
    main()
