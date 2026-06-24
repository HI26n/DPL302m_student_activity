"""
train.py
========
Training script sử dụng Transfer Learning (ResNet18) cho phân loại hành vi sinh viên.

Dữ liệu: dataset/split_by_person/ (chia theo người, tránh leakage)
    - train : aug của p1-p4
    - val   : raw của p1 + p5 (chọn model / early stopping)
    - test  : raw của p1 + p5 (p1 đã thấy, p5 người mới)

Chạy:
    python scripts/model/train.py   # sau split_by_person

Kết quả:
    - Best model: models/best_model.pth (chọn theo val acc)
    - Test metrics: logs/training/final_test_results.json
    - Training curves: logs/training/training_curves.png
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

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
    "model_name": "resnet18",
    "num_classes": 5,
    "input_size": 224,
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "weight_decay": 1e-5,
    "early_stopping_patience": 5,
    "data_root": "dataset/split_by_person",
    "train_dir": "dataset/split_by_person/train",
    "val_dir": "dataset/split_by_person/val",
    "test_dir": "dataset/split_by_person/test",
    "model_dir": "models",
    "log_dir": "logs/training",
    "split_strategy": "by_person: p1-p4 aug train, p1+p5 raw val/test, no raw↔aug leak",
}

CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]

# ============================================================================
# DATA HELPERS
# ============================================================================

def count_split_images(root_dir, classes):
    """Đếm ảnh theo class trong một split."""
    counts = {}
    total = 0
    for cls_name in classes:
        cls_path = os.path.join(root_dir, cls_name)
        if not os.path.exists(cls_path):
            counts[cls_name] = 0
            continue
        n = len([
            f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        counts[cls_name] = n
        total += n
    return counts, total


def validate_data_paths():
    """Kiểm tra thư mục dữ liệu trước khi train."""
    missing = []
    empty_splits = []

    for split_key in ("train_dir", "val_dir", "test_dir"):
        split_path = CONFIG[split_key]
        if not os.path.exists(split_path):
            missing.append(split_path)
            continue
        _, total = count_split_images(split_path, CLASSES)
        if total == 0:
            empty_splits.append(split_path)

    if missing:
        print("❌ Không tìm thấy thư mục dữ liệu:")
        for path in missing:
            print(f"   - {path}")
        print("\n   Chạy pipeline split theo người trước:")
        print("   python scripts/data/augment_data.py")
        print("   python scripts/data/label_persons.py")
        print("   python scripts/data/split_by_person.py\n")
        sys.exit(1)

    if empty_splits:
        print("❌ Split rỗng:")
        for path in empty_splits:
            print(f"   - {path}")
        sys.exit(1)

    print("[Data] Split summary:")
    for split_key, label in (
        ("train_dir", "Train"),
        ("val_dir", "Val"),
        ("test_dir", "Test"),
    ):
        counts, total = count_split_images(CONFIG[split_key], CLASSES)
        print(f"  {label:<6}: {total:>5} ảnh  {counts}")
    print(f"  Strategy: {CONFIG['split_strategy']}")
    print("  (Test chỉ dùng đánh giá cuối, không tham gia chọn model)\n")


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
            root_dir: thư mục gốc (ví dụ: dataset/split_by_person/train)
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
    Test set được load riêng sau khi training xong.
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


def get_test_loader(batch_size=32, num_workers=0):
    """DataLoader cho test set — chỉ dùng đánh giá cuối."""
    test_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_dataset = StudentActivityDataset(
        CONFIG["test_dir"], CLASSES, transforms=test_transforms
    )
    if len(test_dataset) == 0:
        print("[WARNING] Test set rỗng — bỏ qua đánh giá cuối.")
        return None
    return DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def load_best_model(model_name, num_classes, model_path):
    """Tải best checkpoint để đánh giá test."""
    model = create_model(model_name, num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def run_final_test_evaluation(criterion):
    """
    Đánh giá một lần trên test set sau khi training xong.
    Dùng best model đã lưu theo val accuracy.
    """
    best_model_path = os.path.join(CONFIG["model_dir"], "best_model.pth")
    if not os.path.exists(best_model_path):
        print("[WARNING] Không có best model — bỏ qua test evaluation.")
        return None

    test_loader = get_test_loader(batch_size=CONFIG["batch_size"])
    if test_loader is None:
        return None

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION (p1 seen + p5 unseen)")
    print("=" * 60)

    model = load_best_model(
        CONFIG["model_name"], CONFIG["num_classes"], best_model_path
    )
    test_loss, test_acc = validate(model, test_loader, criterion, DEVICE)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    results = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_dir": CONFIG["test_dir"],
        "model_path": best_model_path,
        "note": "Test set = raw images of p1 (seen in train) and p5 (unseen)",
    }
    results_path = os.path.join(CONFIG["log_dir"], "final_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Final test results saved: {results_path}")

    return test_acc


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

    validate_data_paths()

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

    test_acc = run_final_test_evaluation(criterion)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    if test_acc is not None:
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        print("→ Chi tiết: python scripts/model/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
