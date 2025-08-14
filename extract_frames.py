import cv2
import os

def extract_frames(video_path, output_dir, every_n_frames=5):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            filename = os.path.join(output_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(filename, frame)
            frame_id += 1
        count += 1
    cap.release()

# Example: Extract from both datasets
extract_frames("dataset2/normal/video1.mp4", "dataset/NORAML")
extract_frames("dataset2/cashlift/cctv_CAM 5_main_20250524212334.dav", "dataset/cashlifting")
