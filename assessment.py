import cv2
import numpy as np
import torch
import lpips

def calculate_temporal_consistency(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    tc_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, frame)
            tc_value = np.mean(diff)
            tc_values.append(tc_value)
        prev_frame = frame

    cap.release()
    avg_tc = np.mean(tc_values)
    return avg_tc

def calculate_lpips(video_path):
    loss_fn = lpips.LPIPS(net='alex')
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    lpips_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if prev_frame is not None:
            lpips_value = loss_fn(prev_frame, frame)
            lpips_values.append(lpips_value.item())
        prev_frame = frame

    cap.release()
    avg_lpips = np.mean(lpips_values)
    return avg_lpips

def assess_videos(video1_path, video2_path):
    tc_video1 = calculate_temporal_consistency(video1_path)
    tc_video2 = calculate_temporal_consistency(video2_path)
    lpips_video1 = calculate_lpips(video1_path)
    lpips_video2 = calculate_lpips(video2_path)

    return {
        "temporal_consistency": {
            "video1": tc_video1,
            "video2": tc_video2
        },
        "lpips": {
            "video1": lpips_video1,
            "video2": lpips_video2
        }
    }
