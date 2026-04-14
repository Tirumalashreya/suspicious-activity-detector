#extract_frames.py
import cv2
import os
from glob import glob

def extract_all_videos(video_folder, label_name, output_root, max_frames=50):
    video_files = []
    for ext in ('**/*.mp4', '**/*.avi', '**/*.mov', '**/*.mkv'):
        found = glob(os.path.join(video_folder, ext), recursive=True)
        print(f"[DEBUG] Searching in {video_folder} for {ext} → Found {len(found)} files")
        video_files.extend(found)

    if not video_files:
        print(f"[WARNING] No videos found in {video_folder}")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_folder = os.path.join(output_root, label_name, video_name)
        os.makedirs(out_folder, exist_ok=True)
        extract_frames(video_path, out_folder, max_frames)

def extract_frames(video_path, output_dir, max_frames=50):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()

if __name__ == "__main__":
    output_root = "dataset/train"
    os.makedirs(output_root, exist_ok=True)

    # === All known suspicious activity folders ===
    suspicious_dirs = [
        'archive/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Abuse',
        'archive/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arson',
        'archive/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Assault',
        'archive/Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Arrest',
        'archive/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Shoplifting',
        'archive/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Stealing',
        'archive/Anomaly-Videos-Part-4/Anomaly-Videos-Part-4/Vandalism',
        'archive/Burglary',
        'archive/FightingA_Part1',
        'archive/FightingA_Part2',
        'archive/FightingA_Part3',
        'archive/FightingA_Part11',
        'archive/Augmented_Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Abuse'
    ]

    for folder in suspicious_dirs:
        extract_all_videos(folder, 'suspicious', output_root)

    # === Normal activity folder ===
    normal_dir = 'archive/Normal_Videos_for_Event_Recognition'
    extract_all_videos(normal_dir, 'normal', output_root)

    print("\n✅ Frame extraction complete.")