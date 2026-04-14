# train_model.py — Final version with class weighting to fix over-detection of suspicious
# Key fix: class-weighted loss so normal videos are not ignored

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

class LabelSmoothingCrossEntropy(nn.Module):
    """CrossEntropy with label smoothing"""
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

def main():
    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # ─── Config ──────────────────────────────────────────────────────
    data_dir       = 'dataset/train'
    batch_size     = 96               # reduced for stability
    accum_steps    = 4                # effective batch ~384
    epochs         = 30
    lr             = 0.0002
    weight_decay   = 0.02
    patience       = 5
    label_smoothing = 0.1
    num_workers    = 6

    output_model   = "model/suspicious_detector.pt"

    # ─── Data ────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.35, 0.35, 0.25, 0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=None)

    # Class counts & weights
    class_counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in full_dataset.classes}
    print("Class counts:", class_counts)

    total_samples = sum(class_counts.values())
    weight_normal    = total_samples / (2 * class_counts.get('normal', 1))
    weight_suspicious = total_samples / (2 * class_counts.get('suspicious', 1))

    class_weights = torch.tensor([weight_normal, weight_suspicious], dtype=torch.float).to(device)
    print(f"Class weights → normal: {weight_normal:.3f}, suspicious: {weight_suspicious:.3f}")

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}\n")

    # ─── Model ───────────────────────────────────────────────────────
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(0.55),
        nn.Linear(384, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )

    model = model.to(device)

    # ─── Loss & Optimizer ────────────────────────────────────────────
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    # Weighted version (most important change)
    weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Use weighted loss (you can blend if desired)
    loss_fn = weighted_criterion   # ← main fix

    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ─── Training Loop ───────────────────────────────────────────────
    print("Starting training...\n")
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = train_correct = train_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        optimizer.zero_grad(set_to_none=True)

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * accum_steps
            pred = outputs.argmax(dim=1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{train_loss/(i+1):.4f}", acc=f"{100*train_correct/train_total:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc  = 100 * train_correct / train_total

        # Validation
        model.eval()
        val_loss = val_correct = val_total = 0.0

        with torch.no_grad():
            pbar_v = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val  ]")
            for images, labels in pbar_v:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

                pbar_v.set_postfix(loss=f"{val_loss/(pbar_v.n+1):.4f}", acc=f"{100*val_correct/val_total:.2f}%")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc  = 100 * val_correct / val_total

        print(f"Epoch {epoch+1:2d}  Train  {avg_train_loss:.4f} / {avg_train_acc:.2f}%")
        print(f"          Val    {avg_val_loss:.4f} / {avg_val_acc:.2f}%")

        scheduler.step(avg_val_loss)
        print(f"LR: {optimizer.param_groups[0]['lr']:.7f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_model)
            print("  → saved best model")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        if device.type == "mps":
            torch.mps.empty_cache()

    print(f"\nTraining finished in {(time.time()-start_time)/60:.1f} min")
    print(f"Model saved → {output_model}")

if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except:
        pass
    main()