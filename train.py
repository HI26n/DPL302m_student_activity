"""
train.py
========
Training script sử dụng Transfer Learning (ResNet18) cho phân loại hành vi sinh viên.

Chạy:
    python train.py

Kết quả:
    - Model được train trên split_train, validate trên split_val
    - Lưu best model vào models/best_model.pth
    - Tạo training curves và metrics vào logs/
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import cv2
import sys

# ============================================================================
# CONFIG
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

CONFIG = {
    "model_name": "resnet18",  # transfer learning model
    "num_classes": 5,
    "input_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "train_dir": "dataset/split/train",
    "val_dir": "dataset/split/val",
    "model_dir": "models",
    "log_dir": "logs",
}

CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]

# ============================================================================
# DATASET CLASS
# ============================================================================

class StudentActivityDataset(Dataset):
    """
    Custom dataset cho ảnh student activity.
    Đọc ảnh từ thư mục class và apply transforms.
    """

    def __init__(self, root_dir, classes, transforms=None):
        """
        Args:
            root_dir: thư mục gốc (ví dụ: dataset/split/train)
            classes: danh sách tên class
            transforms: augmentation transforms
        """
        self.root_dir = root_dir
        self.classes = classes
        self.transforms = transforms
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.images = []
        self.labels = []

        # Đọc tất cả ảnh từ các folder class
        for cls_name in classes:
            cls_path = os.path.join(root_dir, cls_name)
            if not os.path.exists(cls_path):
                continue
            for img_name in sorted(os.listdir(cls_path)):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(cls_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

        print(f"[Dataset] Loaded {len(self.images)} images from {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # Đọc ảnh bằng OpenCV (BGR) và convert sang RGB
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Failed to read {img_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms nếu có
        if self.transforms:
            image = self.transforms(image)

        return image, label


# ============================================================================
# DATA LOADERS
# ============================================================================

def get_data_loaders(batch_size=32, num_workers=0):
    """
    Tạo DataLoaders cho train và validation.
    """
    # Transforms cho training (có augmentation)
    train_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Transforms cho validation (không augmentation)
    val_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Datasets
    train_dataset = StudentActivityDataset(CONFIG["train_dir"], CLASSES, transforms=train_transforms)
    val_dataset = StudentActivityDataset(CONFIG["val_dir"], CLASSES, transforms=val_transforms)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


# ============================================================================
# MODEL
# ============================================================================

def create_model(model_name="resnet18", num_classes=5):
    """
    Tạo model từ pretrained weights (transfer learning).
    Thay output layer để phù hợp với số class.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Thay output layer
    if "resnet" in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif "efficientnet" in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    print(f"[Model] Created {model_name} with {num_classes} classes")
    return model


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train một epoch.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate trên validation set.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({"loss": f"{total_loss / (pbar.n + 1):.4f}"})

    epoch_loss = total_loss / len(val_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print("TRANSFER LEARNING TRAINING")
    print("=" * 60)
    print(f"Model: {CONFIG['model_name']}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print("=" * 60 + "\n")

    # Create data loaders
    train_loader, val_loader = get_data_loaders(batch_size=CONFIG["batch_size"])

    # Create model
    model = create_model(CONFIG["model_name"], CONFIG["num_classes"])
    model = model.to(DEVICE)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

    # Learning rate scheduler (optional)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # Training loop
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n[Epoch {epoch + 1}/{CONFIG['epochs']}]")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_path = os.path.join(CONFIG["model_dir"], "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✅ Best model saved: {best_model_path} (Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"\n[Early Stopping] No improvement for {CONFIG['early_stopping_patience']} epochs. Stopping.")
                break

        # LR scheduler
        scheduler.step(val_acc)

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(CONFIG["log_dir"], "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n✅ Training history saved: {history_path}")

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(history["val_acc"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(CONFIG["log_dir"], "training_curves.png")
    fig.savefig(curves_path, dpi=150)
    plt.close()
    print(f"✅ Training curves saved: {curves_path}")

    # Save config
    config_path = os.path.join(CONFIG["log_dir"], "training_config.json")
    with open(config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✅ Config saved: {config_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
