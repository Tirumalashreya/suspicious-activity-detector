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

## 📦 Dataset

This project uses the **UCF Anomaly Detection Dataset**:

👉 https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf

### 📥 Setup Dataset

1. Download the dataset from Kaggle
2. Extract it into the project folder as:

```
archive/
```

### Expected structure:

```
archive/
├── Anomaly-Videos-Part-1/
├── Anomaly-Videos-Part-4/
├── Normal_Videos_for_Event_Recognition/
├── Burglary/
├── FightingA_Part1/
...
```

---

## 📁 Dataset Creation (VERY IMPORTANT)

You do NOT need to manually create `normal/` and `suspicious/` folders.

Run:

```bash
python extract_frames.py
```

👉 This script will automatically:

* Read videos from `archive/`
* Categorize them into:

  * `normal`
  * `suspicious`
* Extract frames from each video

---

## 📁 Final Dataset Structure

After running the script, your dataset will look like:

```
dataset/
└── train/
    ├── normal/
    │   ├── video1/
    │   │   ├── frame_000.jpg
    │   │   ├── frame_001.jpg
    │   │   └── ...
    │   └── ...
    │
    └── suspicious/
        ├── video2/
        │   ├── frame_000.jpg
        │   ├── frame_001.jpg
        │   └── ...
        └── ...
```

---

## 🧠 Class Mapping

The model automatically assigns labels based on folder names:

* `normal` → 0
* `suspicious` → 1

Handled internally by `torchvision.datasets.ImageFolder`.

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

---

## 🧠 Step 2: Train Model

```bash
python train_model.py
```

👉 Output:

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

* Dataset is NOT included in this repo (too large)
* You must download it manually from Kaggle
* Training and inference architecture MUST match
* Model file must exist before running app

---

## 🚀 Features

* Real-time detection
* Works on CPU / GPU
* Simple and reproducible pipeline

---

## ❗ Common Errors

### Error: size mismatch

👉 Cause: using wrong model or wrong script
👉 Fix: retrain using `train_model.py`

---

## 👩‍💻 Author

Shreya Svs
