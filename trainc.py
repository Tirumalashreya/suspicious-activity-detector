import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


def main():
    # ================= CONFIG =================
    data_dir = 'dataset/train'
    batch_size = 32
    epochs = 20
    val_split = 0.2
    early_stop_patience = 5

    # ================= DEVICE =================
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
        print("🚀 Using Apple MPS (Metal) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🖥️ Using CPU")

    memory_format = torch.channels_last
    num_workers = min(8, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    # ================= TRANSFORMS =================
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ================= DATASET =================
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    # 🔧 FIX: apply validation transforms correctly
    val_dataset.dataset = datasets.ImageFolder(
        data_dir, transform=val_transforms
    )

    print(f"📊 Dataset split: {train_size} training, {val_size} validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ================= MODEL =================
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2),
    )

    model = model.to(device=device, memory_format=memory_format)

    # ================= OPTIMIZER =================
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )

    # ✅ FIXED: removed unsupported `verbose`
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    # ================= TRAINING =================
    best_val_loss = float("inf")
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"\n🧠 Starting training | Epochs: {epochs}\n")
    start_time = time.time()

    for epoch in range(epochs):
        # -------- TRAIN --------
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=pin_memory)\
                           .contiguous(memory_format=memory_format)
            labels = labels.to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # -------- VALIDATE --------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=pin_memory)\
                               .contiguous(memory_format=memory_format)
                labels = labels.to(device, non_blocking=pin_memory)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total

        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print("=" * 60)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # -------- EARLY STOP --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            os.makedirs("model", exist_ok=True)
            torch.save(model.state_dict(), "model/best_model.pt")
            print("✅ Best model saved")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{early_stop_patience})")

        if epochs_no_improve >= early_stop_patience:
            print("🛑 Early stopping triggered")
            break

    elapsed = time.time() - start_time
    print(f"\n⏱️ Training time: {elapsed/60:.2f} minutes")

    torch.save(model.state_dict(), "model/final_model.pt")

    with open("model/training_history.json", "w") as f:
        json.dump(
            {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "train_accs": train_accs,
                "val_accs": val_accs,
            },
            f,
        )

    print("\n🎉 Training complete!")
    print(f"Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
