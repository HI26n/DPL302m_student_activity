"""
review_data.py
==============
Tạo ảnh grid thumbnail để kiểm tra toàn bộ dataset bằng mắt.
Kết quả lưu vào dataset/review/review_<class>.jpg

Cách chạy:
    python scripts/data/review_data.py

Sau khi chạy:
    - Mở thư mục dataset/review/
    - Xem từng file review_<class>.jpg
    - Tìm ảnh lỗi (chỉ có mặt, không thấy đồ vật, mờ, bị che)
    - Xóa ảnh lỗi thủ công trong dataset/raw/<class>/
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import cv2
import os
import numpy as np

CLASSES    = ["attentive", "phone", "laptop", "sleeping", "distracted"]
THUMB_SIZE = 112   # kích thước mỗi thumbnail
GRID_COLS  = 10    # số cột mỗi hàng
MAX_IMGS   = 500   # giới hạn ảnh mỗi class

# Màu viền từng class (BGR) để dễ phân biệt khi mở nhầm file
COLORS = {
    "attentive":  (34,  139, 34),    # xanh lá
    "phone":      (30,  30,  220),   # đỏ
    "laptop":     (20,  140, 220),   # cam
    "sleeping":   (180, 50,  180),   # tím
    "distracted": (20,  180, 220),   # vàng
}


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".png"))])


def print_summary():
    print("\n  📊 dataset/raw/ hiện tại:")
    total = 0
    for cls in CLASSES:
        count = count_images(f"dataset/raw/{cls}")
        bar   = "█" * (count // 8) + "░" * max(0, 20 - count // 8)
        print(f"    {cls:<15} {count:>4}  {bar}")
        total += count
    print(f"    {'TOTAL':<15} {total:>4}\n")


def make_grid(cls_name):
    folder = f"dataset/raw/{cls_name}"
    files  = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ])[:MAX_IMGS]

    if not files:
        print(f"    [{cls_name}] Chưa có ảnh.")
        return

    color  = COLORS.get(cls_name, (160, 160, 160))
    thumbs = []

    for fname in files:
        path = os.path.join(folder, fname)
        img  = cv2.imread(path)
        if img is None:
            continue
        thumb = cv2.resize(img, (THUMB_SIZE, THUMB_SIZE))
        cv2.rectangle(thumb, (0, 0),
                      (THUMB_SIZE - 1, THUMB_SIZE - 1), color, 3)
        thumbs.append(thumb)

    if not thumbs:
        return

    # Ghép thành grid
    rows = []
    for i in range(0, len(thumbs), GRID_COLS):
        chunk = thumbs[i:i + GRID_COLS]
        while len(chunk) < GRID_COLS:
            chunk.append(np.full(
                (THUMB_SIZE, THUMB_SIZE, 3), 45, dtype=np.uint8))
        rows.append(np.hstack(chunk))
    grid = np.vstack(rows)

    # Header
    header = np.full((38, grid.shape[1], 3), 28, dtype=np.uint8)
    label  = f"{cls_name.upper()}  ({len(thumbs)} anh)"
    cv2.putText(header, label, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2,
                cv2.LINE_AA)
    final = np.vstack([header, grid])

    out = f"dataset/review/review_{cls_name}.jpg"
    cv2.imwrite(out, final)
    print(f"    ✅ {cls_name:<15} {len(thumbs):>4} ảnh  →  {out}")


if __name__ == "__main__":
    os.makedirs("dataset/review", exist_ok=True)
    print_summary()

    print("  🖼  Đang tạo review grid...\n")
    for cls in CLASSES:
        if os.path.exists(f"dataset/raw/{cls}"):
            make_grid(cls)
        else:
            print(f"    [{cls}] Thư mục chưa tồn tại.")

    print("\n  ✅ Xong! Mở thư mục dataset/review/ để kiểm tra.")
    print()
    print("  Tiêu chí ảnh đạt:")
    print("    ✓ Thấy rõ đầu + vai + tay")
    print("    ✓ Thấy đồ vật đặc trưng (điện thoại / laptop / sách)")
    print("    ✓ Không mờ, không bị che quá nhiều")
    print()
    print("  Ảnh lỗi → xóa thủ công trong dataset/raw/<class>/")
    print("  Sau đó chạy: python scripts/data/check_dataset.py\n")
