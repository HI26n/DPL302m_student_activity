"""
split_by_person.py
==================
Chia dataset theo người, TRÁNH leakage ảnh gốc ↔ augment.

Chiến lược:
    - Train : aug của p1, p2, p3, p4
    - Val   : 50% raw của p1 + p5
    - Test  : 50% còn lại raw của p1 + p5

Quan trọng (chống leakage):
    Với p1 (vừa train vừa eval): chia theo raw_idx
    - Một nửa raw_idx → chỉ sinh aug vào train
    - Nửa còn lại   → chỉ raw vào val/test
    → Không có cặp raw/aug cùng nguồn nằm ở 2 tập khác nhau
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import os
import re
import shutil
import random
from collections import defaultdict

CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]
INPUT_BASE = "dataset/augmented"
OUTPUT_BASE = "dataset/split_by_person"
SEED = 42
CLEAN_BEFORE_SPLIT = True
AUGMENT_FACTOR = 2  # khớp augment_data.py
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

TRAIN_PERSONS = ["p1", "p2", "p3", "p4"]
EVAL_PERSONS = ["p1", "p5"]
TRAIN_ONLY_PERSONS = ["p2", "p3", "p4"]
OVERLAP_PERSON = "p1"
UNSEEN_PERSON = "p5"
SEEN_IN_TRAIN = ["p1"]
UNSEEN_IN_TRAIN = ["p5"]

# Tỉ lệ ảnh raw của p1 dùng làm nguồn augment cho train (phần còn lại → val/test)
P1_TRAIN_RAW_RATIO = 0.5

RAW_IDX_RE = re.compile(r"_raw_(\d+)\.", re.IGNORECASE)
AUG_IDX_RE = re.compile(r"_aug_(\d+)\.", re.IGNORECASE)

random.seed(SEED)


def extract_person_id(filename):
    for i in range(1, 6):
        prefix = f"p{i}_"
        if filename.startswith(prefix):
            return f"p{i}"
    return None


def extract_raw_idx(filename):
    m = RAW_IDX_RE.search(filename)
    return int(m.group(1)) if m else None


def extract_aug_source_raw_idx(filename):
    m = AUG_IDX_RE.search(filename)
    if not m:
        return None
    return int(m.group(1)) // AUGMENT_FACTOR


def clear_images(folder):
    if not os.path.exists(folder):
        return 0
    removed = 0
    for fname in os.listdir(folder):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            os.remove(os.path.join(folder, fname))
            removed += 1
    return removed


def split_raw_halves(raw_files):
    """Chia danh sách raw 50/50 cho val và test."""
    items = list(raw_files)
    random.shuffle(items)
    n_val = len(items) // 2
    return items[:n_val], items[n_val:]


def split_p1_by_source(files):
    """
    Chia p1 theo raw_idx — train aug và eval raw không chồng nguồn.
    Returns: (train_aug_files, val_raw_files, test_raw_files)
    """
    p1_raw = [
        f for f in files
        if extract_person_id(f) == OVERLAP_PERSON and "_raw_" in f
    ]
    p1_aug = [
        f for f in files
        if extract_person_id(f) == OVERLAP_PERSON and "_aug_" in f
    ]

    random.shuffle(p1_raw)
    n_train_source = int(len(p1_raw) * P1_TRAIN_RAW_RATIO)
    train_source_raw = p1_raw[:n_train_source]
    eval_source_raw = p1_raw[n_train_source:]

    train_source_indices = {extract_raw_idx(f) for f in train_source_raw}
    eval_source_indices = {extract_raw_idx(f) for f in eval_source_raw}

    train_aug = [
        f for f in p1_aug
        if extract_aug_source_raw_idx(f) in train_source_indices
    ]

    val_raw, test_raw = split_raw_halves(eval_source_raw)

    return train_aug, val_raw, test_raw, train_source_indices, eval_source_indices


def split_class(cls_name):
    input_dir = os.path.join(INPUT_BASE, cls_name)

    if not os.path.exists(input_dir):
        print(f"  [{cls_name}] Chưa có dữ liệu — bỏ qua.")
        return {}

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not files:
        print(f"  [{cls_name}] Thư mục trống — bỏ qua.")
        return {}

    unlabeled = [f for f in files if extract_person_id(f) is None]
    if unlabeled:
        print(
            f"  [{cls_name}] ⚠️  {len(unlabeled)} ảnh chưa có nhãn người "
            f"(chạy scripts/data/label_persons.py trước)"
        )

    train_files = []
    val_files = []
    test_files = []

    # p2, p3, p4: toàn bộ aug vào train
    for fname in files:
        person = extract_person_id(fname)
        if person in TRAIN_ONLY_PERSONS and "_aug_" in fname:
            train_files.append(fname)

    # p1: chia theo raw_idx, tránh leakage
    p1_train_aug, p1_val_raw, p1_test_raw, _, _ = split_p1_by_source(files)
    train_files.extend(p1_train_aug)
    val_files.extend(p1_val_raw)
    test_files.extend(p1_test_raw)

    # p5: toàn bộ raw → val/test
    p5_raw = [
        f for f in files
        if extract_person_id(f) == UNSEEN_PERSON and "_raw_" in f
    ]
    p5_val, p5_test = split_raw_halves(p5_raw)
    val_files.extend(p5_val)
    test_files.extend(p5_test)

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }

    counts = {}
    for split_name, split_files in splits.items():
        out_dir = os.path.join(OUTPUT_BASE, split_name, cls_name)
        os.makedirs(out_dir, exist_ok=True)

        if CLEAN_BEFORE_SPLIT:
            clear_images(out_dir)

        for fname in split_files:
            shutil.copy2(
                os.path.join(input_dir, fname),
                os.path.join(out_dir, fname),
            )

        counts[split_name] = len(split_files)

    return counts


def check_source_leakage():
    """
    Kiểm tra không có raw_idx của p1 vừa trong train (qua aug) vừa trong val/test (raw).
    """
    print(f"\n  {'─' * 65}")
    print(f"  {'KIỂM TRA LEAKAGE (p1 raw_idx)':^65}")
    print(f"  {'─' * 65}")

    total_overlap = 0
    for cls in CLASSES:
        train_dir = os.path.join(OUTPUT_BASE, "train", cls)
        val_dir = os.path.join(OUTPUT_BASE, "val", cls)
        test_dir = os.path.join(OUTPUT_BASE, "test", cls)

        train_aug_indices = set()
        eval_raw_indices = set()

        if os.path.exists(train_dir):
            for fname in os.listdir(train_dir):
                if extract_person_id(fname) != OVERLAP_PERSON or "_aug_" not in fname:
                    continue
                idx = extract_aug_source_raw_idx(fname)
                if idx is not None:
                    train_aug_indices.add(idx)

        for eval_dir in (val_dir, test_dir):
            if not os.path.exists(eval_dir):
                continue
            for fname in os.listdir(eval_dir):
                if extract_person_id(fname) != OVERLAP_PERSON or "_raw_" not in fname:
                    continue
                idx = extract_raw_idx(fname)
                if idx is not None:
                    eval_raw_indices.add(idx)

        overlap = train_aug_indices & eval_raw_indices
        if overlap:
            total_overlap += len(overlap)
            print(f"  ❌ {cls}: {len(overlap)} raw_idx trùng train↔eval")
        else:
            print(f"  ✅ {cls}: không trùng nguồn p1")

    print(f"  {'─' * 65}")
    if total_overlap == 0:
        print("  ✅ Không phát hiện leakage raw↔aug giữa train và val/test\n")
    else:
        print(f"  ❌ Tổng {total_overlap} raw_idx bị trùng — cần kiểm tra lại!\n")

    return total_overlap == 0


def print_summary(all_counts):
    train_persons = ", ".join(TRAIN_PERSONS)
    eval_persons = ", ".join(EVAL_PERSONS)

    print(f"\n  {'═' * 65}")
    print(f"  {'KẾT QUẢ SPLIT DATASET THEO NGƯỜI':^65}")
    print(f"  {'═' * 65}")
    print(f"  {'Class':<15} {'Train':>12} {'Val':>12} {'Test':>12}")
    print(f"  {'─' * 65}")

    totals = defaultdict(int)
    for cls, counts in all_counts.items():
        train = counts.get("train", 0)
        val = counts.get("val", 0)
        test = counts.get("test", 0)
        print(f"  {cls:<15} {train:>12} {val:>12} {test:>12}")
        totals["train"] += train
        totals["val"] += val
        totals["test"] += test

    print(f"  {'─' * 65}")
    grand = totals["train"] + totals["val"] + totals["test"]
    print(
        f"  {'TOTAL':<15} {totals['train']:>12} "
        f"{totals['val']:>12} {totals['test']:>12}"
    )
    print(f"  {'═' * 65}")

    if grand > 0:
        print(f"\n  📊 Cấu trúc tập dữ liệu:")
        print(f"    - Train : aug của {train_persons}")
        print(f"      (p1: chỉ aug từ {int(P1_TRAIN_RAW_RATIO*100)}% raw_idx, không trùng val/test)")
        print(f"    - Val   : 50% raw của {eval_persons}")
        print(f"    - Test  : 50% raw của {eval_persons}")
        print(f"    - Test có: {', '.join(SEEN_IN_TRAIN)} (đã thấy) + "
              f"{', '.join(UNSEEN_IN_TRAIN)} (mới)")
  print(f"\n  → Bước tiếp theo: python scripts/model/train.py")
  print(f"  → Đánh giá cuối : python scripts/model/evaluate.py\n")


if __name__ == "__main__":
    print("\n🔄 Đang chia dataset theo người (chống leakage)...\n")
    print(f"  Train persons : {', '.join(TRAIN_PERSONS)} (aug only)")
    print(f"  Eval persons  : {', '.join(EVAL_PERSONS)} (raw → val + test)")
    print(f"  p1 train raw  : {int(P1_TRAIN_RAW_RATIO*100)}% raw_idx → aug train, "
          f"còn lại → val/test\n")

    all_counts = {}
    for cls in CLASSES:
        print(f"  Xử lý: {cls}")
        counts = split_class(cls)
        if counts:
            all_counts[cls] = counts
            print(
                f"    train={counts.get('train', 0)} | "
                f"val={counts.get('val', 0)} | "
                f"test={counts.get('test', 0)}"
            )

    if all_counts:
        print_summary(all_counts)
        check_source_leakage()
    else:
        print(
            "❌ Không có dữ liệu để split.\n"
            "   Chạy: augment_data.py → label_persons.py → file này.\n"
        )
