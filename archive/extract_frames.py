import os, cv2
from glob import glob

def extract_frames(video_path, output_dir, fps=10):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(int(native_fps / fps), 1)

    idx, saved = 0, 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()

def process_all_videos(dataset_dir, frames_dir="frames", fps=10):
    os.makedirs(frames_dir, exist_ok=True)
    videos = glob(os.path.join(dataset_dir, "**/*.mp4"), recursive=True)
    print(f"🎬 Found {len(videos)} videos.")
    for vid in videos:
        name = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(frames_dir, name)
        if not os.path.exists(out_dir):
            print(f"Extracting {name}...")
            extract_frames(vid, out_dir, fps)
        else:
            print(f"Skipping {name}, already exists.")

if __name__ == "__main__":
    process_all_videos("datasets/archive", "frames", fps=10)
