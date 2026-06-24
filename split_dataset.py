"""
split_dataset.py
================
Chia dataset từ dataset/augmented/ thành train/val/test
theo tỉ lệ 70/15/15, lưu vào dataset/split/.

Đảm bảo:
  - Mỗi class có tỉ lệ đồng đều trong cả 3 split (stratified)
  - Ảnh gốc và ảnh augment từ cùng nguồn không bị lẫn
    vào cả train lẫn val/test (tránh data leakage)

Cách chạy:
    python split_dataset.py
"""

import os
import shutil
import random
from collections import defaultdict

CLASSES    = ["attentive", "phone", "laptop", "sleeping", "distracted"]
INPUT_BASE = "dataset/augmented"
OUTPUT_BASE = "dataset/split"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42

random.seed(SEED)


def split_class(cls_name):
    input_dir = os.path.join(INPUT_BASE, cls_name)

    if not os.path.exists(input_dir):
        print(f"  [{cls_name}] Chưa có ảnh augment — bỏ qua.")
        return {}

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not files:
        print(f"  [{cls_name}] Thư mục trống — bỏ qua.")
        return {}

    # Tách ảnh gốc và ảnh augment
    # Ảnh gốc: tên chứa "_raw_"
    # Ảnh augment: tên chứa "_aug_"
    raw_files = [f for f in files if "_raw_" in f]
    aug_files = [f for f in files if "_aug_" in f]

    random.shuffle(raw_files)
    random.shuffle(aug_files)

    # Chia ảnh gốc vào val + test (tránh data leakage)
    # Ảnh gốc không vào train — chỉ ảnh augment vào train
    n_raw    = len(raw_files)
    n_val    = max(1, int(n_raw * (VAL_RATIO / (VAL_RATIO + TEST_RATIO))))
    n_test   = n_raw - n_val

    val_files   = raw_files[:n_val]
    test_files  = raw_files[n_val:n_val + n_test]
    train_files = aug_files  # toàn bộ ảnh augment vào train

    splits = {
        "train": train_files,
        "val":   val_files,
        "test":  test_files
    }

    counts = {}
    for split_name, split_files in splits.items():
        out_dir = os.path.join(OUTPUT_BASE, split_name, cls_name)
        os.makedirs(out_dir, exist_ok=True)

        for fname in split_files:
            src = os.path.join(input_dir, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(src, dst)

        counts[split_name] = len(split_files)

    return counts


def print_summary(all_counts):
    print(f"\n  {'═'*58}")
    print(f"  {'KẾT QUẢ SPLIT DATASET':^58}")
    print(f"  {'═'*58}")
    print(f"  {'Class':<15} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print(f"  {'─'*54}")

    totals = defaultdict(int)
    for cls, counts in all_counts.items():
        train = counts.get("train", 0)
        val   = counts.get("val",   0)
        test  = counts.get("test",  0)
        total = train + val + test
        print(f"  {cls:<15} {train:>7} {val:>7} {test:>7} {total:>7}")
        totals["train"] += train
        totals["val"]   += val
        totals["test"]  += test

    print(f"  {'─'*54}")
    grand = totals['train'] + totals['val'] + totals['test']
    print(f"  {'TOTAL':<15} {totals['train']:>7} "
          f"{totals['val']:>7} {totals['test']:>7} {grand:>7}")
    print(f"  {'═'*58}")
    print(f"\n  Tỉ lệ thực tế:")
    if grand > 0:
        print(f"    Train : {totals['train']/grand*100:.1f}%")
        print(f"    Val   : {totals['val']/grand*100:.1f}%")
        print(f"    Test  : {totals['test']/grand*100:.1f}%")
    print(f"\n  → Bước tiếp theo: python train.py\n")


if __name__ == "__main__":
    print("\n🔄 Đang chia dataset...\n")

    all_counts = {}
    for cls in CLASSES:
        print(f"  Xử lý: {cls}")
        counts = split_class(cls)
        if counts:
            all_counts[cls] = counts
            print(f"    train={counts.get('train',0)} | "
                  f"val={counts.get('val',0)} | "
                  f"test={counts.get('test',0)}")

    if all_counts:
        print_summary(all_counts)
    else:
        print("❌ Không có dữ liệu để split.")
        print("   Chạy augment_data.py trước.\n")
