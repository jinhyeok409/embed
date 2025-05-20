import cv2
import numpy as np
import posenet
import tensorflow as tf
import time
import os

# 설정
output_dir = 'dataset/fall'  # 또는 'dataset/normal'
os.makedirs(output_dir, exist_ok=True)
seq_length = 90  # 프레임 수
camera_id = 0

with tf.Session() as sess:
    model_cfg, model_outputs = posenet.load_model(101, sess)
    output_stride = model_cfg['output_stride']

    cap = cv2.VideoCapture(camera_id)
    frames = []

    print("데이터 수집 시작...")

    while len(frames) < seq_length:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=0.7125, output_stride=output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=1,
            min_pose_score=0.15)

        keypoint_coords *= output_scale

        if pose_scores[0] == 0.:
            continue

        # 17개의 keypoint (y, x) 좌표 추출
        keypoints = keypoint_coords[0].flatten()
        frames.append(keypoints)

        cv2.imshow('PoseNet', display_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 데이터 저장
    data = np.array(frames)
    timestamp = int(time.time())
    filename = os.path.join(output_dir, f'pose_{timestamp}.npy')
    np.save(filename, data)
    print(f"데이터 저장 완료: {filename}")
