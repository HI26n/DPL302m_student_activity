"""
setup_project.py
================
Chạy MỘT LẦN DUY NHẤT khi bắt đầu project.
Tạo toàn bộ cấu trúc thư mục cần thiết.

Cách chạy:
    python setup_project.py
"""

import os

CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]

folders = (
    ["videos", "dataset/review", "models", "logs"]
    + [f"dataset/raw/{c}"       for c in CLASSES]
    + [f"dataset/augmented/{c}" for c in CLASSES]
    + [f"dataset/split/train/{c}" for c in CLASSES]
    + [f"dataset/split/val/{c}"   for c in CLASSES]
    + [f"dataset/split/test/{c}"  for c in CLASSES]
)

print("Đang tạo cấu trúc thư mục...\n")
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# In cây thư mục
print("✅ Hoàn tất! Cấu trúc project:\n")
for root, dirs, files in os.walk("."):
    dirs[:] = sorted([
        d for d in dirs
        if not d.startswith(".")
        and d not in ("__pycache__", "node_modules")
    ])
    level  = root.replace(".", "").count(os.sep)
    indent = "    " * level
    print(f"{indent}{os.path.basename(root)}/")

print("\n→ Bước tiếp theo: đặt video vào thư mục videos/ rồi chạy extract_frames.py")
