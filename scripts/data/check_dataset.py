"""
check_dataset.py
================
Kiểm tra số lượng và chất lượng dataset trước khi augmentation.
Chạy sau khi đã review và xóa ảnh lỗi xong.

Cách chạy:
    python scripts/data/check_dataset.py
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import os
import cv2
import numpy as np
from collections import defaultdict

CLASSES    = ["attentive", "phone", "laptop", "sleeping", "distracted"]
TARGET     = 150   # ảnh gốc tối thiểu mỗi class
IMG_SIZE   = 224   # kích thước chuẩn sau extract


def count_images(folder):
    if not os.path.exists(folder):
        return []
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ]


def check_image_sizes(folder, files, sample=30):
    """Kiểm tra tất cả ảnh có đúng 224x224 không."""
    wrong = []
    for fname in files[:sample]:
        img = cv2.imread(os.path.join(folder, fname))
        if img is None:
            wrong.append(f"{fname} (không đọc được)")
            continue
        h, w = img.shape[:2]
        if h != IMG_SIZE or w != IMG_SIZE:
            wrong.append(f"{fname} ({w}x{h})")
    return wrong


def main():
    print("\n" + "═"*52)
    print(f"  {'KIỂM TRA DATASET TRƯỚC KHI AUGMENTATION':^50}")
    print("═"*52)

    counts   = {}
    all_ok   = True
    issues   = defaultdict(list)

    for cls in CLASSES:
        folder = f"dataset/raw/{cls}"
        files  = count_images(folder)
        count  = len(files)
        counts[cls] = count

        # Kiểm tra kích thước ảnh
        if count > 0:
            wrong_size = check_image_sizes(folder, files)
            if wrong_size:
                issues[cls].extend([f"Sai kích thước: {f}" for f in wrong_size[:3]])

        # Đánh giá
        if count >= TARGET:
            status = "✅"
        elif count >= TARGET // 2:
            status = "⚠️ "
            all_ok = False
        else:
            status = "❌"
            all_ok = False

        print(f"\n  {status} {cls.upper()}")
        print(f"     Số ảnh : {count} / {TARGET} cần thiết")
        if count < TARGET:
            print(f"     Cần thêm: {TARGET - count} ảnh")
        if issues[cls]:
            for issue in issues[cls]:
                print(f"     ⚠️  {issue}")

    # Tổng kết
    total = sum(counts.values())
    print(f"\n{'─'*52}")
    print(f"  Tổng ảnh gốc : {total}")
    print(f"  Dự kiến sau augment (×5): ~{total * 5}")
    print(f"  Train/Val/Test (70/15/15): "
          f"~{int(total*5*0.7)} / ~{int(total*5*0.15)} / ~{int(total*5*0.15)}")

    print(f"\n{'═'*52}")
    if all_ok:
        print("  ✅ Dataset sẵn sàng cho augmentation!")
        print("  → Bước tiếp theo: python scripts/data/augment_data.py")
    else:
        print("  ⚠️  Một số class chưa đủ ảnh.")
        print("  → Quay thêm video, thêm vào VIDEO_MAP,")
        print("     rồi chạy lại: python scripts/data/extract_frames.py")
    print("═"*52 + "\n")


if __name__ == "__main__":
    main()
