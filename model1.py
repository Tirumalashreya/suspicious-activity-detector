# model_loader.py
# Updated to match train_anomaly_detector.py architecture (512 → 256 → 2)

import torch
import torch.nn as nn
from torchvision import models

def load_model():
    # Create the same ResNet50 structure used during training
    model = models.resnet50(pretrained=False)

    # Freeze backbone (same as training)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 (same as training)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # ─── THIS MUST EXACTLY MATCH train_anomaly_detector.py ───
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )
    # ──────────────────────────────────────────────────────────

    # Load the trained weights
    try:
        # Try loading in order of preference
        model_loaded = False
        
        # 1. Try best_model.pt (from trainc.py)
        try:
            checkpoint = torch.load("model/best_model.pt", map_location="cpu")
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Best model loaded (Epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%)")
            model_loaded = True
        except FileNotFoundError:
            pass
        
        # 2. Try final_model.pt (from trainc.py)
        if not model_loaded:
            try:
                state_dict = torch.load("model/final_model.pt", map_location="cpu")
                model.load_state_dict(state_dict)
                print("✅ Final model loaded successfully.")
                model_loaded = True
            except FileNotFoundError:
                pass
        
        # 3. Try suspicious_detector.pt (your existing model - may not match architecture)
        if not model_loaded:
            try:
                state_dict = torch.load("model/suspicious_detector.pt", map_location="cpu")
                model.load_state_dict(state_dict)
                print("⚠️ WARNING: Loaded suspicious_detector.pt - this may not match the new architecture!")
                print("💡 Recommendation: Re-train using trainc.py for best results")
                model_loaded = True
            except (FileNotFoundError, RuntimeError):
                pass
        
        if not model_loaded:
            raise FileNotFoundError("No model file found! Please train the model first using trainc.py")
            
    except RuntimeError as e:
        print("❌ Error loading model weights:")
        print(e)
        print("\n⚠️ Possible cause: Model architecture mismatch")
        print("💡 Solution: Your existing model was trained with a different architecture.")
        print("   Delete model/suspicious_detector.pt and re-train using trainc.py")
        raise

    model.eval()
    return model