"""
setup_project.py
================
Chạy MỘT LẦN khi bắt đầu project — tạo cấu trúc thư mục.

Cách chạy (từ thư mục gốc):
    python setup_project.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.paths import CLASSES, PROJECT_ROOT

FOLDERS = (
    ["videos", "models", "logs", "docs/reports"]
    + ["logs/eda", "logs/training", "logs/evaluation"]
    + ["scripts/data", "scripts/model", "config"]
    + [f"dataset/raw/{c}" for c in CLASSES]
    + [f"dataset/augmented/{c}" for c in CLASSES]
    + [f"dataset/review"]
    + [f"dataset/split_by_person/train/{c}" for c in CLASSES]
    + [f"dataset/split_by_person/val/{c}" for c in CLASSES]
    + [f"dataset/split_by_person/test/{c}" for c in CLASSES]
)

print("Đang tạo cấu trúc thư mục...\n")
for folder in FOLDERS:
    os.makedirs(PROJECT_ROOT / folder, exist_ok=True)

SKIP = {".git", "__pycache__", "node_modules", "dataset", "logs", "videos", "models"}
print("✅ Hoàn tất! Cấu trúc project:\n")
print(f"{PROJECT_ROOT.name}/")
print("├── README.md")
print("├── requirements.txt")
print("├── setup_project.py")
print("├── config/")
print("├── scripts/")
print("│   ├── data/          ← thu thập & xử lý dữ liệu")
print("│   └── model/         ← EDA, train, evaluate")
print("├── docs/              ← hướng dẫn chi tiết")
print("├── videos/            ← đặt video quay (.mp4/.mov)")
print("├── dataset/")
print("│   ├── raw/")
print("│   ├── augmented/")
print("│   ├── review/")
print("│   └── split_by_person/")
print("├── models/            ← best_model.pth")
print("└── logs/")
print("    ├── eda/")
print("    ├── training/")
print("    └── evaluation/")
print("\n→ Tiếp theo: xem README.md và chạy pipeline.")
