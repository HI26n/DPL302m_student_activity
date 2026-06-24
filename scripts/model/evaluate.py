"""
evaluate.py
===========
Đánh giá model đã train trên test set.

Kết quả lưu tại logs/evaluation/:
    - evaluation_results.json
    - classification_report.txt
    - confusion_matrix.png

Chạy sau train.py:
    python scripts/model/evaluate.py
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import seaborn as sns
from tqdm import tqdm
import json

# ============================================================================
# CONFIG
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

CONFIG = {
    "model_path": "models/best_model.pth",
    "model_name": "resnet18",
    "num_classes": 5,
    "input_size": 224,
    "batch_size": 32,
    "test_dir": "dataset/split_by_person/test",
    "log_dir": "logs/evaluation",
}

CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]

# ============================================================================
# DATASET CLASS (reuse from train.py)
# ============================================================================

class StudentActivityDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, classes, transforms=None):
        self.root_dir = root_dir
        self.classes = classes
        self.transforms = transforms
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.images = []
        self.labels = []

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

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Failed to read {img_path}")
            return None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms(image)

        return image, label


# ============================================================================
# MODEL
# ============================================================================

def load_model(model_path, model_name="resnet18", num_classes=5):
    """
    Tải model từ checkpoint.
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Thay output layer
    if "resnet" in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif "efficientnet" in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Load weights
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()

    print(f"[Model] Loaded from {model_path}")
    return model


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(model, test_loader, device, classes):
    """
    Evaluate model trên test set.
    """
    all_labels = []
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, f1, report, cm, all_labels, all_preds


def plot_confusion_matrix(cm, classes, output_path):
    """
    Vẽ confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Check if model exists
    if not os.path.exists(CONFIG["model_path"]):
        print(f"❌ Model not found: {CONFIG['model_path']}")
        print("   Please train the model first: python scripts/model/train.py")
        return

    # Load model
    model = load_model(
        CONFIG["model_path"],
        CONFIG["model_name"],
        CONFIG["num_classes"],
    )

    # Create test dataset
    test_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = StudentActivityDataset(
        CONFIG["test_dir"],
        CLASSES,
        transforms=test_transforms,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    # Evaluate
    print("\n[Evaluating]")
    accuracy, f1, report, cm, all_labels, all_preds = evaluate(
        model,
        test_loader,
        DEVICE,
        CLASSES,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Print classification report
    report_df = pd.DataFrame(report).transpose()
    print(report_df)

    # Save results
    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "classification_report": report,
    }

    results_path = os.path.join(CONFIG["log_dir"], "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved: {results_path}")

    # Save classification report
    report_path = os.path.join(CONFIG["log_dir"], "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted F1-Score: {f1:.4f}\n\n")
        f.write(str(report_df))
    print(f"✅ Classification report saved: {report_path}")

    # Plot confusion matrix
    cm_path = os.path.join(CONFIG["log_dir"], "confusion_matrix.png")
    plot_confusion_matrix(cm, CLASSES, cm_path)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
