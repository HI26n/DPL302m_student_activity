"""
label_persons.py
================
Gán nhãn người (p1-p5) NHẤT QUÁN trên mọi class.

Vấn đề cũ: K-Means chạy riêng từng class → p1 ở attentive khác p1 ở sleeping.
Cách mới:
    1. Gom tất cả ảnh raw (mọi class) → K-Means toàn cục (k=5)
    2. Lan nhãn sang ảnh augment tương ứng (cùng raw index)

Pipeline:
    python scripts/data/augment_data.py
    python scripts/data/label_persons.py
    python scripts/data/split_by_person.py
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import os
import re
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

INPUT_BASE = "dataset/augmented"
NUM_PERSONS = 5
AUGMENT_FACTOR = 2  # phải khớp augment_data.py
CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
PERSON_PREFIX_RE = re.compile(r"^p[1-5]_")
RAW_NAME_RE = re.compile(r"^(.+)_raw_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)
AUG_NAME_RE = re.compile(r"^(.+)_aug_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

print("🔄 Đang tải mô hình ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def strip_person_prefix(filename):
  """Bỏ tiền tố p1_-p5_ nếu có."""
  if PERSON_PREFIX_RE.match(filename):
    return PERSON_PREFIX_RE.sub("", filename, count=1)
  return filename


def add_person_prefix(person_id, filename):
  """Thêm tiền tố người, tránh trùng nếu đã có."""
  base = strip_person_prefix(filename)
  return f"{person_id}_{base}"


def strip_all_existing_labels():
  """Xóa nhãn người cũ để gán lại toàn cục."""
  stripped = 0
  for cls in CLASSES:
    input_dir = os.path.join(INPUT_BASE, cls)
    if not os.path.exists(input_dir):
      continue
    for fname in os.listdir(input_dir):
      if not fname.lower().endswith(IMAGE_EXTENSIONS):
        continue
      if not PERSON_PREFIX_RE.match(fname):
        continue
      src = os.path.join(input_dir, fname)
      dst = os.path.join(input_dir, strip_person_prefix(fname))
      if src != dst:
        os.rename(src, dst)
        stripped += 1
  return stripped


def extract_features(img_path):
  try:
    img = Image.open(img_path).convert("RGB")
    img_t = transform(img).unsqueeze(0)
    with torch.no_grad():
      features = model(img_t)
    return features.squeeze().numpy()
  except Exception as e:
    print(f"  Lỗi đọc ảnh {img_path}: {e}")
    return None


def collect_raw_images():
  """Thu thập mọi ảnh raw từ tất cả class."""
  entries = []
  for cls in CLASSES:
    input_dir = os.path.join(INPUT_BASE, cls)
    if not os.path.exists(input_dir):
      continue
    for fname in sorted(os.listdir(input_dir)):
      if not fname.lower().endswith(IMAGE_EXTENSIONS):
        continue
      base = strip_person_prefix(fname)
      m = RAW_NAME_RE.match(base)
      if not m:
        continue
      class_name, raw_idx, _ = m.group(1), int(m.group(2)), m.group(3)
      if class_name != cls:
        continue
      entries.append({
        "class": cls,
        "raw_idx": raw_idx,
        "path": os.path.join(input_dir, fname),
        "fname": fname,
      })
  return entries


def cluster_globally(raw_entries):
  """K-Means trên toàn bộ ảnh raw — cùng 1 người = cùng 1 cluster."""
  filepaths = []
  features_list = []
  meta = []

  print(f"\n  Trích xuất đặc trưng cho {len(raw_entries)} ảnh raw (mọi class)...")
  for entry in raw_entries:
    feat = extract_features(entry["path"])
    if feat is None:
      continue
    filepaths.append(entry["path"])
    features_list.append(feat)
    meta.append(entry)

  if len(features_list) < NUM_PERSONS:
    raise ValueError(
      f"Chỉ có {len(features_list)} ảnh raw hợp lệ, cần ít nhất {NUM_PERSONS}."
    )

  print(f"  K-Means toàn cục: {NUM_PERSONS} người trên {len(features_list)} ảnh...")
  kmeans = KMeans(n_clusters=NUM_PERSONS, random_state=42, n_init=10)
  labels = kmeans.fit_predict(features_list)

  # (class, raw_idx) -> person_id
  person_map = {}
  for entry, label in zip(meta, labels):
    key = (entry["class"], entry["raw_idx"])
    person_map[key] = f"p{label + 1}"

  return person_map


def apply_person_labels(person_map):
  """Đổi tên raw + aug theo person_map."""
  renamed_raw = 0
  renamed_aug = 0
  skipped_aug = 0

  for cls in CLASSES:
    input_dir = os.path.join(INPUT_BASE, cls)
    if not os.path.exists(input_dir):
      continue

    for fname in sorted(os.listdir(input_dir)):
      if not fname.lower().endswith(IMAGE_EXTENSIONS):
        continue

      base = strip_person_prefix(fname)
      path = os.path.join(input_dir, fname)

      m_raw = RAW_NAME_RE.match(base)
      if m_raw:
        raw_idx = int(m_raw.group(2))
        person = person_map.get((cls, raw_idx))
        if not person:
          continue
        new_name = add_person_prefix(person, base)
        new_path = os.path.join(input_dir, new_name)
        if path != new_path:
          os.rename(path, new_path)
          renamed_raw += 1
        continue

      m_aug = AUG_NAME_RE.match(base)
      if m_aug:
        aug_idx = int(m_aug.group(2))
        raw_idx = aug_idx // AUGMENT_FACTOR
        person = person_map.get((cls, raw_idx))
        if not person:
          skipped_aug += 1
          continue
        new_name = add_person_prefix(person, base)
        new_path = os.path.join(input_dir, new_name)
        if path != new_path:
          os.rename(path, new_path)
          renamed_aug += 1

  return renamed_raw, renamed_aug, skipped_aug


def print_person_consistency_report(person_map):
  """Kiểm tra mỗi người có xuất hiện ở nhiều class không."""
  by_person = defaultdict(lambda: defaultdict(int))
  for (cls, _raw_idx), person in person_map.items():
    by_person[person][cls] += 1

  print(f"\n  {'─' * 60}")
  print(f"  {'BÁO CÁO NHẤT QUÁN NHÃN NGƯỜI (trên ảnh raw)':^60}")
  print(f"  {'─' * 60}")
  print(f"  {'Person':<8} {'Classes':>8}  Phân bố theo class")
  print(f"  {'─' * 60}")

  for person in sorted(by_person.keys()):
    class_counts = by_person[person]
    n_classes = len(class_counts)
    detail = ", ".join(f"{c}={n}" for c, n in sorted(class_counts.items()))
    flag = "✅" if n_classes >= 3 else "⚠️ "
    print(f"  {flag} {person:<6} {n_classes:>8}  {detail}")

  print(f"  {'─' * 60}")
  print("  (Mỗi person nên xuất hiện ở nhiều class — cùng 1 người thật)\n")


def cluster_and_rename():
  stripped = strip_all_existing_labels()
  if stripped:
    print(f"  Đã gỡ {stripped} nhãn người cũ (gán lại toàn cục).\n")

  raw_entries = collect_raw_images()
  if not raw_entries:
    print("❌ Không tìm thấy ảnh raw. Chạy augment_data.py trước.")
    return

  person_map = cluster_globally(raw_entries)
  renamed_raw, renamed_aug, skipped_aug = apply_person_labels(person_map)

  print(f"\n  ✅ Đổi tên: {renamed_raw} raw, {renamed_aug} aug")
  if skipped_aug:
    print(f"  ⚠️  Bỏ qua {skipped_aug} ảnh aug (không map được raw)")

  print_person_consistency_report(person_map)


if __name__ == "__main__":
  cluster_and_rename()
  print("🎉 HOÀN TẤT! Nhãn người đã đồng nhất trên mọi class.")
  print("👉 Bước tiếp theo: python scripts/data/split_by_person.py")
