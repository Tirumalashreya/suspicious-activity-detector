# app.py — Suspicious Activity Detector
# Supports: Webcam (live) OR Video file input
# Usage examples:
#   python app.py                               → webcam (default)
#   python app.py --source video --path yourvideo.mp4  → video file
#   python app.py --source webcam --camera-index 1     → specific camera

import argparse
import cv2
import torch
import subprocess
import threading
import queue
import warnings
import time
from model_loader import load_model

# Suppress noisy deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision.*")

LABELS = ['Normal activity', 'Suspicious activity']

# Device: CPU is usually more stable for single-frame inference on M4
# (change to "mps" if you want to test GPU acceleration)
device = torch.device("cpu")
print(f"Using device for inference: {device}")

# Load model once
model = load_model()
model.to(device)
model.eval()

# TTS via macOS 'say' command
speech_queue = queue.Queue()

def _tts_worker():
    while True:
        text = speech_queue.get()
        try:
            subprocess.call(['say', '-v', 'Samantha', text])
        except Exception as e:
            print(f"TTS error: {e}")
        finally:
            speech_queue.task_done()

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

def play_alert():
    speak("Warning! Suspicious activity detected")

# Preprocess frame
def preprocess(frame, size=(224, 224)):
    img = cv2.resize(frame, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Suspicious Activity Detector")
    parser.add_argument('--source', choices=['webcam', 'video'], default='webcam',
                        help="Input source: 'webcam' (default) or 'video'")
    parser.add_argument('--path', type=str, default=None,
                        help="Path to video file (required if --source=video)")
    parser.add_argument('--camera-index', type=int, default=0,
                        help="Webcam camera index (default: 0)")

    args = parser.parse_args()

    # ─── DEBUG PRINTS ────────────────────────────────────────────────
    print("\n=== DEBUG: Parsed arguments ===")
    print(f"source       : {args.source}")
    print(f"path         : {args.path}")
    print(f"camera_index : {args.camera_index}")
    print("=============================\n")
    # ─────────────────────────────────────────────────────────────────

    # Validate arguments
    if args.source == 'video' and not args.path:
        print("Error: --path is required when --source=video")
        return

    # Open source
    cap = None
    is_webcam = (args.source == 'webcam')

    if is_webcam:
        print(f"Trying to open webcam (index {args.camera_index})...")
        for api in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(args.camera_index, api)
            if cap.isOpened():
                print(f"Webcam opened successfully (backend: {api})")
                break
        if not cap or not cap.isOpened():
            print("ERROR: Could not open webcam. Check camera permissions.")
            return
    else:
        print(f"Opening video file: {args.path}")
        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file: {args.path}")
            return

    # Video properties
    fps_input = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30.0
    if fps_input <= 0:
        fps_input = 30.0

    max_duration_sec = 300 if is_webcam else float('inf')  # 5 min only for webcam
    start_time = time.time()
    frame_count = 0
    prev_time = time.time()

    print(f"Starting detection — {'Webcam' if is_webcam else 'Video file'} mode")
    print(f"Press 'q' to quit | Max duration (webcam only): 5 min\n")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_duration_sec:
                print("Reached maximum duration (5 minutes). Stopping.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_webcam:
                    print("Warning: Failed to grab frame from webcam — retrying...")
                    time.sleep(0.1)
                    continue
                else:
                    print("End of video reached.")
                    break

            frame_count += 1

            # FPS calculation
            now = time.time()
            current_fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            try:
                input_tensor = preprocess(frame).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    if isinstance(output, (list, tuple)):
                        output = output[0]
                    pred = torch.argmax(output, dim=1)[0] if output.dim() > 1 else torch.argmax(output)
                    label = LABELS[pred.item()]
            except Exception as e:
                print(f"Inference error: {e}")
                label = "Inference error"

            # Visualize
            color = (0, 0, 255) if label == "Suspicious activity" else (0, 255, 0)
            cv2.putText(frame, f"{label}  FPS: {current_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if label == "Suspicious activity":
                print(f"[{time.strftime('%H:%M:%S')}] Suspicious activity → alerting!")
                play_alert()

            cv2.imshow("Suspicious Activity Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User quit.")
                break
            elif key == ord(' '):
                print("Manual alert test...")
                play_alert()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_queue.join(timeout=8)
        print("Detection stopped.")

if __name__ == "__main__":
    main()