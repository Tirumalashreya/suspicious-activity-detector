# 🚨 Suspicious Activity Detection System

## 📌 Overview

This project detects suspicious activities from video streams using a deep learning model (ResNet50).

---

## ⚠️ IMPORTANT (STRICT USAGE)

This project follows **ONE FIXED PIPELINE ONLY**.

👉 Use ONLY these files:

* `extract_frames.py`
* `train_model.py`
* `model_loader.py`
* `app.py`

❌ Do NOT use:

* `app1.py`
* `train1.py`
* `trainc.py`
* `train_model_old.py`
* `model1.py`

These are experimental and may break the pipeline.

---

## 🧠 Pipeline

```
Videos → Frames → Dataset → Train Model → Detection
```

---

## 📁 Project Structure

```
.
├── extract_frames.py        # Step 1: Create dataset
├── train_model.py          # Step 2: Train model
├── model_loader.py         # Step 3: Load trained model
├── app.py                  # Step 4: Run detection

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

👉 Converts videos → images
👉 Creates dataset inside `dataset/train`

---

## 🧠 Step 2: Train Model

```bash
python train_model.py
```

👉 What happens:

* Loads images from dataset
* Uses ResNet50 (pretrained)
* Trains classifier (normal vs suspicious)
* Saves model:

```
model/suspicious_detector.pt
```

---

## 🎥 Step 3: Run Detection

### Webcam

```bash
python app.py
```

### Video file

```bash
python app.py --source video --path video.mp4
```

---

## 🔄 How Prediction Works

1. Frame is captured
2. Resized to 224×224
3. Converted to tensor
4. Passed to model
5. Output:

   * Normal activity
   * Suspicious activity

---

## ⚠️ Important Notes

* Training and inference architecture MUST match
* Dataset should be balanced
* Model file must exist before running app

---

## 🚀 Features

* Real-time detection
* Works on CPU / GPU
* Simple and reproducible pipeline

---

## ❗ Common Errors

### Error: size mismatch

👉 Cause: using wrong model file or wrong script
👉 Fix: retrain using `train_model.py`

---

## 👩‍💻 Author

Shreya Svs
