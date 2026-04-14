# train1.py — GPU + FAST, MATCHES model_loader.py (2048 → 384 → 128 → 2)

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

def main():
    # ================= DEVICE =================
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ MPS (Apple GPU) detected and enabled!")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")

    # MPS stability (safe)
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # ================= CONFIG =================
    data_dir = "dataset/train"
    batch_size = 128
    accum_steps = 2
    epochs = 25
    lr = 3e-4
    weight_decay = 0.01
    num_workers = 8
    output_model = "model/suspicious_detector.pt"

    print(f"Device: {device}")
    print(f"Effective batch size: {batch_size * accum_steps}\n")

    # ================= TRANSFORMS =================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.4)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    # ================= DATA =================
    full_ds = datasets.ImageFolder(data_dir, transform=None)

    indices = torch.randperm(len(full_ds)).tolist()
    split = int(0.8 * len(indices))

    train_ds = Subset(full_ds, indices[:split])
    val_ds   = Subset(full_ds, indices[split:])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # ================= MODEL =================
    model = models.resnet50(pretrained=True)

    # Freeze backbone (same as loader)
    for p in model.parameters():
        p.requires_grad = False

    # 🔒 EXACT SAME HEAD AS model_loader.py
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 384),
        nn.ReLU(inplace=True),
        nn.Dropout(0.55),
        nn.Linear(384, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 2)
    )

    model.to(device)

    # ================= TRAINING =================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scaler = GradScaler()
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device_type="mps"):
                out = model(x)
                loss = criterion(out, y) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            preds = out.argmax(1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            pbar.set_postfix(
                acc=f"{100 * correct / total:.2f}%",
                gpu=f"{torch.mps.current_allocated_memory() // 1024**2}MB"
            )

        # ================= VALIDATION =================
        model.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                out = model(x)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"\nEpoch {epoch+1} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_model)
            print("✅ Best model saved\n")

        if device.type == "mps":
            torch.mps.empty_cache()

    print("🚀 Training complete.")

if __name__ == "__main__":
    main()
