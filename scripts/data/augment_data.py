import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import cv2
import os
import random
import shutil
import numpy as np

CLASSES = [
    "attentive",
    "phone",
    "laptop",
    "sleeping",
    "distracted"
]

INPUT_BASE = "dataset/raw"
OUTPUT_BASE = "dataset/augmented"

AUGMENT_FACTOR = 2
SEED = 42
TARGET_SIZE = 224
CLEAN_BEFORE_AUGMENT = True
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(SEED)
np.random.seed(SEED)

def aug_flip(img):
    if random.random() < 0.5:
        return cv2.flip(img, 1)
    return img

def aug_rotation(img):
    angle = random.uniform(-5, 5)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_perspective(img):
    h, w = img.shape[:2]
    shift = int(min(h, w) * 0.04)

    src = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])

    dst = np.float32([
        [random.randint(0, shift), random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), random.randint(0, shift)],
        [random.randint(0, shift), h - 1 - random.randint(0, shift)],
        [w - 1 - random.randint(0, shift), h - 1 - random.randint(0, shift)]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def aug_crop_resize(img):
    h, w = img.shape[:2]
    margin = int(min(h, w) * 0.05)

    x1 = random.randint(0, margin)
    y1 = random.randint(0, margin)
    x2 = w - random.randint(0, margin)
    y2 = h - random.randint(0, margin)

    crop = img[y1:y2, x1:x2]

    return cv2.resize(
        crop,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_AREA
    )

def aug_brightness_contrast(img):
    alpha = random.uniform(0.85, 1.20)
    beta = random.randint(-20, 20)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def aug_color_temperature(img):
    img = img.astype(np.float32)
    mode = random.choice(["warm", "cool", "neutral"])

    if mode == "warm":
        img[:, :, 2] *= random.uniform(1.00, 1.10)
        img[:, :, 0] *= random.uniform(0.90, 1.00)
    elif mode == "cool":
        img[:, :, 0] *= random.uniform(1.00, 1.08)
        img[:, :, 2] *= random.uniform(0.92, 1.00)

    return np.clip(img, 0, 255).astype(np.uint8)

def aug_blur(img):
    mode = random.choice(["gaussian", "motion"])

    if mode == "gaussian":
        k = random.choice([3, 5])
        return cv2.GaussianBlur(img, (k, k), 0)

    k = random.randint(3, 5)
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

def aug_noise(img):
    sigma = random.uniform(2, 6)
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def aug_jpeg_compression(img):
    quality = random.randint(70, 95)

    success, encoded = cv2.imencode(
        ".jpg",
        img,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    )

    if not success:
        return img

    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)

def augment_one(img):

    if random.random() < 0.50:
        img = aug_flip(img)

    if random.random() < 0.30:
        img = aug_rotation(img)

    if random.random() < 0.20:
        img = aug_perspective(img)

    if random.random() < 0.20:
        img = aug_crop_resize(img)

    if random.random() < 0.70:
        img = aug_brightness_contrast(img)

    if random.random() < 0.50:
        img = aug_color_temperature(img)

    if random.random() < 0.25:
        img = aug_blur(img)

    if random.random() < 0.20:
        img = aug_noise(img)

    if random.random() < 0.30:
        img = aug_jpeg_compression(img)

    if img.shape[:2] != (TARGET_SIZE, TARGET_SIZE):
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    return img

def clear_images(folder):
    """Xóa ảnh cũ trong folder trước khi augment lại."""
    if not os.path.exists(folder):
        return 0
    removed = 0
    for fname in os.listdir(folder):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            os.remove(os.path.join(folder, fname))
            removed += 1
    return removed

def augment_class(class_name):

    input_dir = os.path.join(INPUT_BASE, class_name)
    output_dir = os.path.join(OUTPUT_BASE, class_name)

    os.makedirs(output_dir, exist_ok=True)

    if CLEAN_BEFORE_AUGMENT:
        removed = clear_images(output_dir)
        if removed:
            print(f"  [{class_name}] Đã xóa {removed} ảnh cũ trong {output_dir}")

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    if not files:
        return 0

    total_saved = 0

    for idx, fname in enumerate(files):
        shutil.copy2(
            os.path.join(input_dir, fname),
            os.path.join(output_dir, f"{class_name}_raw_{idx:04d}.jpg")
        )
        total_saved += 1

    aug_idx = 0

    for fname in files:

        img = cv2.imread(os.path.join(input_dir, fname))

        if img is None:
            continue

        for _ in range(AUGMENT_FACTOR):

            aug = augment_one(img.copy())

            cv2.imwrite(
                os.path.join(
                    output_dir,
                    f"{class_name}_aug_{aug_idx:04d}.jpg"
                ),
                aug
            )

            aug_idx += 1
            total_saved += 1

    return total_saved

def print_summary(results):

    total = sum(results.values())

    print("\n" + "=" * 50)
    print("AUGMENTATION SUMMARY")
    print("=" * 50)

    for cls, count in results.items():
        print(f"{cls:<15}: {count}")

    print("-" * 50)
    print(f"TOTAL          : {total}")

if __name__ == "__main__":

    results = {}

    for cls in CLASSES:
        results[cls] = augment_class(cls)

    print_summary(results)
