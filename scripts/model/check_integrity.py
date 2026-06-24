"""
check_integrity.py
==================
Kiểm tra leakage và phân bố class trên dataset/split_by_person/.

Chạy:
    python scripts/model/check_integrity.py
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import os
import re
from collections import defaultdict

from config.paths import CLASSES, IMAGE_EXTENSIONS, SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, p

RAW_IDX_RE = re.compile(r"_raw_(\d+)\.", re.IGNORECASE)
AUG_IDX_RE = re.compile(r"_aug_(\d+)\.", re.IGNORECASE)
AUGMENT_FACTOR = 2
OVERLAP_PERSON = "p1"


def extract_person_id(filename):
    for i in range(1, 6):
        prefix = f"p{i}_"
        if filename.startswith(prefix):
            return f"p{i}"
    return None


def collect_split_files(split_dir):
    files = defaultdict(list)
    for cls in CLASSES:
        cls_path = os.path.join(p(split_dir), cls)
        if not os.path.exists(cls_path):
            continue
        for img in os.listdir(cls_path):
            if img.lower().endswith(IMAGE_EXTENSIONS):
                files[img].append(f"{split_dir.name}/{cls}")
    return files


def check_filename_leakage():
    print("\n" + "=" * 70)
    print("DATA INTEGRITY CHECK — split_by_person")
    print("=" * 70)

    train_files = collect_split_files(SPLIT_TRAIN)
    val_files = collect_split_files(SPLIT_VAL)
    test_files = collect_split_files(SPLIT_TEST)

    print("\n📋 Dataset counts:")
    print(f"  Train : {len(train_files)}")
    print(f"  Val   : {len(val_files)}")
    print(f"  Test  : {len(test_files)}")

    print("\n🔍 Filename leakage:")
    overlap_tv = set(train_files) & set(val_files)
    overlap_tt = set(train_files) & set(test_files)
    overlap_vt = set(val_files) & set(test_files)

    for label, overlap in (
        ("TRAIN ↔ VAL", overlap_tv),
        ("TRAIN ↔ TEST", overlap_tt),
        ("VAL ↔ TEST", overlap_vt),
    ):
        if overlap:
            print(f"  ❌ {label}: {len(overlap)} file trùng tên")
            for img in list(overlap)[:3]:
                print(f"     - {img}")
        else:
            print(f"  ✅ Không trùng {label}")

    return not (overlap_tv or overlap_tt or overlap_vt)


def check_p1_source_leakage():
    """Kiểm tra p1 aug train vs p1 raw val/test cùng raw_idx."""
    print("\n🔍 p1 source leakage (raw_idx):")

    train_aug_idx = set()
    eval_raw_idx = set()

    for cls in CLASSES:
        train_cls = os.path.join(p(SPLIT_TRAIN), cls)
        if os.path.exists(train_cls):
            for fname in os.listdir(train_cls):
                if extract_person_id(fname) != OVERLAP_PERSON or "_aug_" not in fname:
                    continue
                m = AUG_IDX_RE.search(fname)
                if m:
                    train_aug_idx.add((cls, int(m.group(1)) // AUGMENT_FACTOR))

        for split in (SPLIT_VAL, SPLIT_TEST):
            eval_cls = os.path.join(p(split), cls)
            if not os.path.exists(eval_cls):
                continue
            for fname in os.listdir(eval_cls):
                if extract_person_id(fname) != OVERLAP_PERSON or "_raw_" not in fname:
                    continue
                m = RAW_IDX_RE.search(fname)
                if m:
                    eval_raw_idx.add((cls, int(m.group(1))))

    overlap = train_aug_idx & eval_raw_idx
    if overlap:
        print(f"  ❌ {len(overlap)} raw_idx p1 trùng giữa train aug và val/test raw")
        for item in list(overlap)[:5]:
            print(f"     - class={item[0]}, raw_idx={item[1]}")
        return False

    print("  ✅ Không trùng nguồn p1 giữa train aug và eval raw")
    return True


def check_class_distribution():
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    for split_name, split_dir in (
        ("train", SPLIT_TRAIN),
        ("val", SPLIT_VAL),
        ("test", SPLIT_TEST),
    ):
        print(f"\n{split_name.upper()}:")
        total = 0
        for cls in CLASSES:
            cls_path = os.path.join(p(split_dir), cls)
            count = 0
            if os.path.exists(cls_path):
                count = len([
                    f for f in os.listdir(cls_path)
                    if f.lower().endswith(IMAGE_EXTENSIONS)
                ])
            total += count
            print(f"  {cls:12s}: {count:4d}")
        print(f"  {'TOTAL':12s}: {total:4d}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n🔍 Checking data integrity...\n")
    ok_names = check_filename_leakage()
    ok_source = check_p1_source_leakage()
    check_class_distribution()

    if ok_names and ok_source:
        print("\n✅ Data looks good!\n")
    else:
        print("\n❌ Issues detected — chạy lại: python scripts/data/split_by_person.py\n")
