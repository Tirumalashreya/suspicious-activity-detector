# app1.py — Suspicious Activity Detector
# Compatible with train_anomaly_detector.py architecture
# Supports: Webcam (live) OR Video file input
# Usage examples:
#   python app1.py                               → webcam (default)
#   python app1.py --source video --path yourvideo.mp4  → video file
#   python app1.py --source webcam --camera-index 1     → specific camera

import argparse
import cv2
import torch
import subprocess
import threading
import queue
import warnings
import time
import numpy as np
from model1 import load_model

# Suppress noisy deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchvision.*")

LABELS = ['Normal activity', 'Suspicious activity']

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple MPS (Metal) for inference")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("🖥️ Using CPU for inference")

# Load model once
print("Loading model...")
model = load_model()
model.to(device)
model.eval()
print("Model ready!")

# TTS via macOS 'say' command
speech_queue = queue.Queue()
last_alert_time = 0
ALERT_COOLDOWN = 5  # seconds between alerts

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
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        speak("Warning! Suspicious activity detected")
        last_alert_time = current_time

def preprocess(frame, size=(224, 224)):
    """
    Preprocess frame to match training pipeline:
    - Resize to 224x224
    - Convert BGR to RGB
    - Normalize with ImageNet mean and std
    """
    # Resize
    img = cv2.resize(frame, size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    img = img.astype("float32") / 255.0
    
    # Apply ImageNet normalization (CRITICAL: must match training)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # Convert to tensor and change dimension order
    tensor = torch.from_numpy(img).permute(2, 0, 1)
    
    return tensor

def get_prediction(frame):
    """
    Get prediction with proper error handling and confidence score
    """
    try:
        input_tensor = preprocess(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            
            # Handle different output formats
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            # Get probabilities
            probs = torch.softmax(output, dim=1)[0]
            
            # Get prediction
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            label = LABELS[pred_idx]
            
        return label, confidence
        
    except Exception as e:
        print(f"⚠️ Inference error: {e}")
        return "Error", 0.0

def main():
    parser = argparse.ArgumentParser(description="Suspicious Activity Detector")
    parser.add_argument('--source', choices=['webcam', 'video'], default='webcam',
                        help="Input source: 'webcam' (default) or 'video'")
    parser.add_argument('--path', type=str, default=None,
                        help="Path to video file (required if --source=video)")
    parser.add_argument('--camera-index', type=int, default=0,
                        help="Webcam camera index (default: 0)")
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help="Minimum confidence for alerts (default: 0.7)")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("📹 Suspicious Activity Detector")
    print("="*60)
    print(f"Source: {args.source}")
    if args.source == 'video':
        print(f"Video path: {args.path}")
    else:
        print(f"Camera index: {args.camera_index}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print("="*60 + "\n")

    # Validate arguments
    if args.source == 'video' and not args.path:
        print("❌ Error: --path is required when --source=video")
        return

    # Open source
    cap = None
    is_webcam = (args.source == 'webcam')

    if is_webcam:
        print(f"Opening webcam (index {args.camera_index})...")
        for api in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(args.camera_index, api)
            if cap.isOpened():
                print(f"✅ Webcam opened (backend: {api})")
                break
        if not cap or not cap.isOpened():
            print("❌ ERROR: Could not open webcam. Check camera permissions.")
            return
    else:
        print(f"Opening video file: {args.path}")
        cap = cv2.VideoCapture(args.path)
        if not cap.isOpened():
            print(f"❌ ERROR: Could not open video file: {args.path}")
            return

    # Video properties
    fps_input = cap.get(cv2.CAP_PROP_FPS) if not is_webcam else 30.0
    if fps_input <= 0:
        fps_input = 30.0

    max_duration_sec = 300 if is_webcam else float('inf')  # 5 min for webcam
    start_time = time.time()
    frame_count = 0
    prev_time = time.time()
    
    # Stats tracking
    suspicious_count = 0
    total_frames = 0

    print(f"\n🎬 Starting detection...")
    print("Press 'q' to quit | Press SPACE for manual alert test")
    if is_webcam:
        print("⏱️ Max duration: 5 minutes\n")

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_duration_sec:
                print("\n⏱️ Reached maximum duration (5 minutes). Stopping.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_webcam:
                    print("⚠️ Warning: Failed to grab frame — retrying...")
                    time.sleep(0.1)
                    continue
                else:
                    print("\n🎬 End of video reached.")
                    break

            frame_count += 1
            total_frames += 1

            # FPS calculation
            now = time.time()
            current_fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
            prev_time = now

            # Get prediction
            label, confidence = get_prediction(frame)

            # Track statistics
            if label == "Suspicious activity" and confidence >= args.confidence_threshold:
                suspicious_count += 1

            # Visualize
            color = (0, 0, 255) if label == "Suspicious activity" else (0, 255, 0)
            
            # Main label with confidence
            cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Stats
            cv2.putText(frame, f"Suspicious: {suspicious_count}/{total_frames}", 
                       (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Alert if suspicious and above threshold
            if label == "Suspicious activity" and confidence >= args.confidence_threshold:
                print(f"[{time.strftime('%H:%M:%S')}] 🚨 SUSPICIOUS (conf: {confidence*100:.1f}%)")
                play_alert()
                # Add warning indicator
                cv2.putText(frame, "⚠️ ALERT!", 
                           (frame.shape[1] - 200, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            cv2.imshow("Suspicious Activity Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 User quit.")
                break
            elif key == ord(' '):
                print("🔊 Manual alert test...")
                play_alert()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        speech_queue.join(timeout=8)
        
        # Final stats
        print("\n" + "="*60)
        print("📊 Detection Statistics")
        print("="*60)
        print(f"Total frames processed: {total_frames}")
        print(f"Suspicious detections: {suspicious_count}")
        if total_frames > 0:
            print(f"Suspicious rate: {(suspicious_count/total_frames)*100:.2f}%")
        print(f"Duration: {elapsed:.1f} seconds")
        print("="*60)
        print("✅ Detection stopped.")

if __name__ == "__main__":
    main()