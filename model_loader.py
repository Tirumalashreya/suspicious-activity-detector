# model_loader.py
# Updated to match the training architecture (384 → 128 → 2 head)

import torch
import torch.nn as nn
from torchvision import models

def load_model():
    # Create the same ResNet50 structure used during training
    model = models.resnet50(pretrained=False)  # pretrained=False because we load full weights anyway

    # Freeze backbone (same as training)
    for param in model.parameters():
        param.requires_grad = False

    # ─── THIS MUST EXACTLY MATCH THE STRUCTURE IN train_model.py ───
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(0.55),
        nn.Linear(384, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )
    # ────────────────────────────────────────────────────────────────

    # Load the trained weights
    try:
        state_dict = torch.load("model/suspicious_detector.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except RuntimeError as e:
        print("Error loading state_dict:")
        print(e)
        print("\nPossible cause: checkpoint was saved with a different fc head structure.")
        print("Solution: delete model/suspicious_detector.pt and re-run training.")
        raise

    model.eval()
    return model