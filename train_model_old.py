import os
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

def main():
    # === Config ===
    data_dir = 'dataset/train'
    batch_size = 32
    epochs = 10
    use_subset = False       # Set to True for quick testing
    subset_size = 1000       # Number of images to use if use_subset = True

    # === Device (prefer Apple MPS on Apple Silicon) ===
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("🚀 Using Apple MPS (Metal) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("🖥️ Using CPU")

    # Performance-related settings
    memory_format = torch.channels_last  # often faster for conv nets on accelerators
    num_workers = min(8, (os.cpu_count() or 1))
    pin_memory = True if device.type == "cuda" else False  # pin_memory helps with CUDA; no benefit for MPS

    # === Transformations ===
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # === Load dataset ===
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

    # Optionally use a subset for fast debugging
    if use_subset:
        dataset = Subset(dataset, range(subset_size))
        print(f"⚠️ Using a subset of {subset_size} images for fast training.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # === Load ResNet50 with latest pretrained weights ===
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor

    # Replace final layer for binary classification
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)  # Binary classification: normal vs suspicious
    )

    # Move model to device and use channels-last memory format for conv performance
    model = model.to(device=device, memory_format=memory_format)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # === Training Loop ===
    print(f"\n🧠 Starting training on {len(dataset)} images | "
          f"Batch size: {batch_size} | Epochs: {epochs}\n")

    train_start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(dataloader):
            # Move inputs to device and convert to channels-last contiguous format (can help performance)
            images = images.to(device=device, non_blocking=pin_memory).contiguous(memory_format=memory_format)
            labels = labels.to(device=device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 5 == 0:
                print(
                    f"[Epoch {epoch+1}] "
                    f"[Batch {batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} complete — Avg Loss: {avg_loss:.4f}\n")

    # Ensure all accelerator operations are finished before timing
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            # requires PyTorch with MPS support that exposes torch.mps.synchronize()
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
    except Exception:
        pass

    train_end_time = time.time()
    elapsed = train_end_time - train_start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"⏱️ Training time: {int(hrs)}h {int(mins)}m {secs:.2f}s")

    # === Save the trained model ===
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), "model/suspicious_detector.pt")

    print("✅ Model trained and saved to model/suspicious_detector.pt")


if __name__ == '__main__':
    # On Windows or when freezing executables, freeze_support() is required for multiprocessing
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()
