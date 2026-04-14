# 🚨 Suspicious Activity Detection System

## 📌 Overview

This project detects suspicious activities from video streams using deep learning (ResNet50).

It supports:

* Webcam-based real-time detection
* Video file analysis
* Audio alerts for suspicious events

---

## 🧠 How it works

1. Convert videos → frames
2. Train CNN model on images
3. Use trained model for real-time detection

---

## 📁 Project Structure

```
.
├── extract_frames.py
├── train_model.py
├── model_loader.py
├── app.py
├── app1.py
├── dataset/
│   └── train/
│       ├── normal/
│       └── suspicious/
├── model/
│   └── suspicious_detector.pt
```

---

## ⚙️ Setup

```bash
pip install torch torchvision opencv-python numpy
```

---

## 🎬 Step 1: Extract Frames

```bash
python extract_frames.py
```

---

## 🧠 Step 2: Train Model

```bash
python train_model.py
```

---

## 🎥 Step 3: Run Detection

### Webcam

```bash
python app1.py
```

### Video file

```bash
python app1.py --source video --path video.mp4
```

---

## 🔊 Features

* Real-time detection
* Confidence scoring
* Audio alerts
* FPS tracking
* Works on CPU / GPU / Apple MPS

---

## ⚠️ Notes

* Ensure dataset is balanced (normal vs suspicious)
* Model accuracy depends on training data quality
* Architecture in `model_loader.py` must match training

---

## 🚀 Future Improvements

* Use video-based models (LSTM / 3D CNN)
* Add object detection (YOLO)
* Deploy as web app / CCTV system
