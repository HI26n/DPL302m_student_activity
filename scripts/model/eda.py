"""
eda.py
======
Khảo sát dữ liệu hiện tại cho pipeline student_activity.

Chạy:
    python scripts/model/eda.py

Kết quả:
    - Phân tích dataset/raw, dataset/augmented, dataset/split_by_person
    - Phân tích cân bằng class, split ratio và các vấn đề imbalance
    - Kiểm tra chất lượng ảnh raw (shape, brightness, contrast, màu)
    - Kiểm tra duplicate và data leakage giữa split
    - Lưu báo cáo và hình ảnh vào logs/eda/
"""

import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'bootstrap.py'))

import sys
import hashlib
import os
import math
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

CLASSES = [
    "attentive",
    "phone",
    "laptop",
    "sleeping",
    "distracted",
]

EXTENSIONS = (".jpg", ".jpeg", ".png")
DATA_ROOTS = {
    "raw": "dataset/raw",
    "augmented": "dataset/augmented",
    "split_train": "dataset/split_by_person/train",
    "split_val": "dataset/split_by_person/val",
    "split_test": "dataset/split_by_person/test",
}

LOG_DIR = "logs/eda"
SAMPLE_SIZE = 80
EXPECTED_SHAPE = (224, 224)


def list_images(folder):
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS)
    ])


def count_images():
    records = []
    for stage, base_folder in DATA_ROOTS.items():
        for cls in CLASSES:
            folder = os.path.join(base_folder, cls)
            files = list_images(folder)
            records.append({
                "stage": stage,
                "class": cls,
                "count": len(files),
                "folder": folder,
            })
    return pd.DataFrame(records)


def analyze_image_stats(folder, sample_size=SAMPLE_SIZE):
    files = list_images(folder)
    if not files:
        return pd.DataFrame(columns=["filename", "width", "height", "brightness", "contrast", "r_mean", "g_mean", "b_mean"])

    if sample_size is not None:
        files = files[:sample_size]
    rows = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        b_mean, g_mean, r_mean = [float(np.mean(img[:, :, i])) for i in range(3)]
        rows.append({
            "filename": os.path.basename(path),
            "width": w,
            "height": h,
            "brightness": brightness,
            "contrast": contrast,
            "r_mean": r_mean,
            "g_mean": g_mean,
            "b_mean": b_mean,
        })

    return pd.DataFrame(rows)


def summary_by_stage(df):
    return df.pivot(index="class", columns="stage", values="count").fillna(0).astype(int)


def compute_split_ratios(summary_df):
    split_cols = [c for c in summary_df.columns if c.startswith("split_")]
    if not split_cols:
        return pd.DataFrame()
    totals = summary_df[split_cols].sum(axis=1).replace(0, 1)
    return summary_df[split_cols].div(totals, axis=0).round(3)


def compute_balance_warnings(series):
    if series.empty:
        return []
    max_count = int(series.max())
    min_count = int(series.min())
    if min_count == 0:
        return ["Một hoặc nhiều class raw đang thiếu ảnh hoàn toàn."]
    ratio = max_count / min_count
    if ratio >= 3:
        return [f"Imbalance mạnh: class lớn nhất gấp {ratio:.1f} lần class nhỏ nhất."]
    if ratio >= 2:
        return [f"Imbalance vừa: class lớn nhất gấp {ratio:.1f} lần class nhỏ nhất."]
    return []


def hash_file(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicate_files(paths):
    groups = defaultdict(list)
    for path in paths:
        groups[hash_file(path)].append(path)
    return {h: sorted(files) for h, files in groups.items() if len(files) > 1}


def plot_distribution(df, title, out_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("class")
    ax.set_ylabel("image count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(df, x_col, y_col, title, out_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df[x_col], df[y_col], alpha=0.7, s=20)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histogram(df, value_col, title, out_path, bins=20, color="tab:blue"):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    df[value_col].hist(bins=bins, ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_shape_summary(df, out_path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["width"], df["height"], alpha=0.7, s=20)
    ax.set_title("Kích thước ảnh mẫu trong dataset/raw")
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_brightness_summary(df, out_path):
    if df.empty:
        return
    mean_stats = df.groupby("class")["brightness"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_stats.plot(kind="bar", color="tab:orange", ax=ax)
    ax.set_title("Brightness trung bình theo class (raw sample)")
    ax.set_xlabel("class")
    ax.set_ylabel("mean brightness")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_color_summary(df, out_path):
    if df.empty:
        return
    mean_stats = df.groupby("class")[["r_mean", "g_mean", "b_mean"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_stats.plot(kind="bar", ax=ax)
    ax.set_title("Giá trị kênh màu trung bình theo class (raw sample)")
    ax.set_xlabel("class")
    ax.set_ylabel("mean intensity")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def compute_summary_stats(df, group_cols, value_cols):
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(group_cols)[value_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    return agg


def compute_augmentation_ratio(df):
    if "augmented" not in df.columns or "raw" not in df.columns:
        return pd.Series(dtype=float)
    ratio = df["augmented"].astype(float) / df["raw"].replace(0, np.nan)
    return ratio.round(2)


def compute_class_imbalance_metrics(counts_series):
    """
    Tính Gini index, entropy, và gợi ý loss function.
    Gini: 0 = cân bằng, 1 = toàn part có 1 class
    Entropy: 0 = cân bằng hoàn hảo (đều), cao = mất cân bằng
    """
    if counts_series.empty or counts_series.sum() == 0:
        return {"gini": 0, "entropy": 0, "imbalance_ratio": 1.0, "recommendation": "No data"}

    total = float(counts_series.sum())
    proportions = counts_series.astype(float) / total

    # Gini index
    gini = 1.0 - np.sum(proportions ** 2)

    # Entropy (normalized)
    eps = 1e-10
    entropy = -np.sum(proportions * np.log(proportions + eps))
    max_entropy = np.log(len(proportions))
    entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0

    # Imbalance ratio
    imbalance_ratio = float(counts_series.max() / counts_series.min()) if counts_series.min() > 0 else np.inf

    # Recommendation
    if imbalance_ratio < 1.5:
        rec = "Cân bằng tốt → dùng standard CrossEntropyLoss"
    elif imbalance_ratio < 2.5:
        rec = "Imbalance vừa → dùng weighted CrossEntropyLoss hoặc focal loss"
    else:
        rec = "Imbalance mạnh → dùng weighted CrossEntropyLoss + augmentation"

    return {
        "gini": round(gini, 4),
        "entropy_normalized": round(entropy_normalized, 4),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "recommendation": rec,
    }


def compute_model_readiness(counts_df):
    """
    Gợi ý cấu hình training dựa trên dataset size và cân bằng.
    """
    train_counts = counts_df[counts_df["stage"] == "split_train"]["count"]
    total_train = int(train_counts.sum())

    # Gợi ý batch size
    if total_train < 500:
        batch_size = 8
    elif total_train < 2000:
        batch_size = 16
    elif total_train < 5000:
        batch_size = 32
    else:
        batch_size = 64

    # Gợi ý epochs
    if total_train < 1000:
        epochs = 50
    elif total_train < 5000:
        epochs = 30
    else:
        epochs = 20

    # Gợi ý learning rate
    lr = 0.0001 if batch_size >= 32 else 0.001

    # Gợi ý metrics
    metrics = ["accuracy", "precision", "recall", "f1"]

    return {
        "total_train_images": total_train,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "optimizer": "Adam",
        "metrics": metrics,
        "augmentation": "RandAugment hoặc AutoAugment đã áp dụng",
    }


def plot_brightness_contrast_boxplot(df, out_path):
    """
    Vẽ boxplot brightness và contrast theo class.
    """
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df.boxplot(column="brightness", by="class", ax=axes[0])
    axes[0].set_title("Brightness distribution by class (raw)")
    axes[0].set_xlabel("class")
    axes[0].set_ylabel("brightness")
    plt.sca(axes[0])
    plt.xticks(rotation=45)

    df.boxplot(column="contrast", by="class", ax=axes[1])
    axes[1].set_title("Contrast distribution by class (raw)")
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("contrast")
    plt.sca(axes[1])
    plt.xticks(rotation=45)

    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def check_split_naming():
    records = []
    for split in ["split_train", "split_val", "split_test"]:
        for cls in CLASSES:
            files = list_images(os.path.join(DATA_ROOTS[split], cls))
            raw_names = [f for f in files if "_raw_" in os.path.basename(f)]
            aug_names = [f for f in files if "_aug_" in os.path.basename(f)]
            records.append({
                "split": split,
                "class": cls,
                "total": len(files),
                "raw_names": len(raw_names),
                "aug_names": len(aug_names),
                "other_names": len(files) - len(raw_names) - len(aug_names),
            })
    return pd.DataFrame(records)


def hash_file(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicate_files(paths):
    groups = defaultdict(list)
    for path in paths:
        groups[hash_file(path)].append(path)
    return {h: sorted(files) for h, files in groups.items() if len(files) > 1}


def find_raw_duplicates():
    duplicates = {}
    for cls in CLASSES:
        files = list_images(os.path.join(DATA_ROOTS["raw"], cls))
        if len(files) < 2:
            continue
        duplicates[cls] = find_duplicate_files(files)
    return duplicates


def find_cross_class_duplicates():
    all_files = []
    for cls in CLASSES:
        all_files.extend(list_images(os.path.join(DATA_ROOTS["raw"], cls)))
    groups = find_duplicate_files(all_files)
    cross = {h: paths for h, paths in groups.items() if len({os.path.dirname(p) for p in paths}) > 1}
    return cross


def make_sample_grid(class_name, out_path, stage="raw", n_images=16):
    base = DATA_ROOTS.get(stage)
    if base is None:
        return False
    folder = os.path.join(base, class_name)
    files = list_images(folder)[:n_images]
    if not files:
        return False

    thumbs = []
    thumb_size = 120
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        thumbs.append(img)

    if not thumbs:
        return False

    cols = int(math.sqrt(len(thumbs)))
    cols = max(1, cols)
    rows = math.ceil(len(thumbs) / cols)

    canvas = np.ones((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8) * 255
    for idx, thumb in enumerate(thumbs):
        r = idx // cols
        c = idx % cols
        y0 = r * thumb_size
        x0 = c * thumb_size
        canvas[y0:y0 + thumb_size, x0:x0 + thumb_size] = thumb

    plt.figure(figsize=(cols * 1.5, rows * 1.5))
    plt.imshow(canvas)
    plt.axis("off")
    plt.title(f"Sample {stage} images: {class_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def collect_folder_stats(stage, cls_name):
    folder = os.path.join(DATA_ROOTS.get(stage, ""), cls_name)
    files = list_images(folder)
    wrong_shape = []
    unreadable = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            unreadable.append(os.path.basename(path))
            continue
        if img.shape[:2] != EXPECTED_SHAPE:
            wrong_shape.append((os.path.basename(path), img.shape[:2]))
    return {
        "class": cls_name,
        "stage": stage,
        "count": len(files),
        "unreadable": len(unreadable),
        "wrong_shape": len(wrong_shape),
        "sample_unreadable": unreadable[:5],
        "sample_wrong_shape": wrong_shape[:5],
    }


def check_split_leakage():
    warnings = []
    overlap = []
    unexpected = []

    for cls in CLASSES:
        sets = {
            split: set(os.path.basename(p) for p in list_images(os.path.join(DATA_ROOTS[split], cls)))
            for split in ["split_train", "split_val", "split_test"]
        }

        for a, b in [("split_train", "split_val"), ("split_train", "split_test"), ("split_val", "split_test")]:
            common = sets[a] & sets[b]
            if common:
                overlap.append((cls, a, b, sorted(list(common))[:10]))

        unexpected_train = [f for f in sets["split_train"] if "_aug_" not in f]
        if unexpected_train:
            unexpected.append((cls, "train", unexpected_train[:10]))

        unexpected_val_test = [f for f in (sets["split_val"] | sets["split_test"]) if "_raw_" not in f]
        if unexpected_val_test:
            unexpected.append((cls, "val/test", sorted(unexpected_val_test)[:10]))

    if overlap:
        warnings.append(("overlap", overlap))
    if unexpected:
        warnings.append(("unexpected_naming", unexpected))
    return warnings


def print_summary_table(df):
    if df.empty:
        return
    print("\n  Class balance / count summary")
    print(df.to_string())


def print_split_ratios(df):
    if df.empty:
        return
    ratios = compute_split_ratios(df)
    print("\n  Split ratios (train/val/test) per class")
    print(ratios.to_string())


def save_report(lines, report_path):
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    report_lines = ["EDA REPORT", "==========", ""]

    print("[EDA] Thu thập số lượng ảnh theo class...")
    counts_df = count_images()
    counts_df.to_csv(os.path.join(LOG_DIR, "eda_image_counts.csv"), index=False)

    summary = summary_by_stage(counts_df)
    summary.to_csv(os.path.join(LOG_DIR, "eda_summary_by_stage.csv"))
    print_summary_table(summary)
    report_lines.append("Class count summary:")
    report_lines.extend(summary.to_string().splitlines())
    report_lines.append("")

    plot_distribution(
        summary,
        "Phân phối số lượng ảnh theo class và giai đoạn",
        os.path.join(LOG_DIR, "eda_distribution_by_stage.png"),
    )

    split_summary = summary[[c for c in summary.columns if c.startswith("split_")]].copy()
    split_summary.to_csv(os.path.join(LOG_DIR, "eda_split_summary.csv"))
    print_split_ratios(split_summary)
    report_lines.append("Split ratios:")
    report_lines.extend(compute_split_ratios(split_summary).to_string().splitlines())
    report_lines.append("")

    augmentation_ratio = compute_augmentation_ratio(summary)
    if not augmentation_ratio.empty:
        augment_df = augmentation_ratio.rename("aug_to_raw_ratio").to_frame()
        augment_df.to_csv(os.path.join(LOG_DIR, "eda_augmentation_ratio.csv"))
        report_lines.append("Augmentation ratio per class:")
        report_lines.extend(augment_df.to_string().splitlines())
        report_lines.append("")

    naming_summary = check_split_naming()
    naming_summary.to_csv(os.path.join(LOG_DIR, "eda_split_naming_summary.csv"), index=False)
    report_lines.append("Split naming pattern summary saved.")
    report_lines.append("")

    raw_balance_warnings = compute_balance_warnings(summary["raw"])
    if raw_balance_warnings:
        for msg in raw_balance_warnings:
            print(f"  ⚠️  {msg}")
            report_lines.append(msg)
    else:
        report_lines.append("Raw class balance looks reasonable.")
    report_lines.append("")

    print("\n[EDA] Kiểm tra hình dạng và thống kê màu sắc ảnh raw...")
    stats_frames = []
    for cls in CLASSES:
        stats = analyze_image_stats(os.path.join(DATA_ROOTS["raw"], cls), sample_size=None)
        if not stats.empty:
            stats["class"] = cls
            stats_frames.append(stats)

    if stats_frames:
        stats_df = pd.concat(stats_frames, ignore_index=True)
        stats_df.to_csv(os.path.join(LOG_DIR, "eda_raw_image_stats.csv"), index=False)
        plot_shape_summary(stats_df, os.path.join(LOG_DIR, "eda_raw_shape_scatter.png"))
        plot_brightness_summary(stats_df, os.path.join(LOG_DIR, "eda_raw_brightness_by_class.png"))
        plot_color_summary(stats_df, os.path.join(LOG_DIR, "eda_raw_color_by_class.png"))
        plot_histogram(stats_df, "brightness", "Brightness distribution (raw)", os.path.join(LOG_DIR, "eda_raw_brightness_hist.png"))
        plot_histogram(stats_df, "contrast", "Contrast distribution (raw)", os.path.join(LOG_DIR, "eda_raw_contrast_hist.png"), color="tab:green")

        shape_summary = stats_df.groupby(["width", "height"]).size().reset_index(name="count")
        shape_summary.to_csv(os.path.join(LOG_DIR, "eda_raw_shape_summary.csv"), index=False)

        raw_color_stats = compute_summary_stats(stats_df, ["class"], ["brightness", "contrast", "r_mean", "g_mean", "b_mean"])
        raw_color_stats.to_csv(os.path.join(LOG_DIR, "eda_raw_color_stats.csv"))

        plot_brightness_contrast_boxplot(stats_df, os.path.join(LOG_DIR, "eda_raw_brightness_contrast_boxplot.png"))

        report_lines.append("Raw image stats saved and plotted.")
        report_lines.append("Raw shape and color summary statistics saved.")
    else:
        print("  Không tìm thấy ảnh raw để phân tích.")
        report_lines.append("No raw images found for stats.")
    report_lines.append("")

    print("[EDA] Kiểm tra chất lượng ảnh và shape across folders...")
    quality_records = []
    for stage in DATA_ROOTS:
        for cls in CLASSES:
            quality_records.append(collect_folder_stats(stage, cls))
    quality_df = pd.DataFrame(quality_records)
    quality_df.to_csv(os.path.join(LOG_DIR, "eda_quality_summary.csv"), index=False)
    print(quality_df[["stage", "class", "count", "unreadable", "wrong_shape"]].to_string(index=False))
    report_lines.append("Quality summary:")
    report_lines.extend(quality_df[["stage", "class", "count", "unreadable", "wrong_shape"]].to_string(index=False).splitlines())
    report_lines.append("")

    print("[EDA] Kiểm tra duplicate ảnh raw...")
    raw_dupes = find_raw_duplicates()
    cross_dupes = find_cross_class_duplicates()
    if raw_dupes:
        report_lines.append("Raw duplicates detected within classes:")
        for cls, groups in raw_dupes.items():
            if groups:
                report_lines.append(f"  {cls}: {len(groups)} duplicate groups")
                for group in list(groups.values())[:3]:
                    report_lines.append("    " + ", ".join(os.path.basename(p) for p in group[:5]))
    else:
        report_lines.append("No duplicate raw files found within classes.")

    if cross_dupes:
        report_lines.append("Cross-class duplicate raw images found:")
        for group in list(cross_dupes.values())[:3]:
            report_lines.append("    " + ", ".join(os.path.basename(p) for p in group[:5]))
    else:
        report_lines.append("No duplicate raw files found across classes.")
    report_lines.append("")

    print("\n[EDA] Kiểm tra data leakage giữa train/val/test...")
    leakage = check_split_leakage()
    if leakage:
        print("  ⚠️  Phát hiện vấn đề data leakage / tên file bất thường:")
        report_lines.append("Split leakage or naming issues:")
        for kind, items in leakage:
            print(f"    - {kind}:")
            report_lines.append(f"  {kind}:")
            for item in items:
                print(f"      {item}")
                report_lines.append(str(item))
    else:
        print("  ✅ Không tìm thấy overlap train/val/test hoặc tên file không hợp lệ.")
        report_lines.append("No split leakage or unexpected file naming detected.")
    report_lines.append("")

    print("\n[EDA] Tạo sample grid cho raw và augmented...")
    for stage in ["raw", "augmented"]:
        for cls in CLASSES:
            out_path = os.path.join(LOG_DIR, f"eda_sample_grid_{stage}_{cls}.png")
            created = make_sample_grid(cls, out_path, stage=stage, n_images=16)
            print(f"  {stage}/{cls}: {'created' if created else 'skipped (no images)'}")

    report_lines.append("Sample grids created for raw and augmented if images exist.")
    report_lines.append("")

    print("\n[EDA] Phân tích imbalance class và khuyến nghị mô hình...")
    train_counts = summary["split_train"]
    imbalance_metrics = compute_class_imbalance_metrics(train_counts)
    report_lines.append("Class imbalance metrics (train set):")
    for key, val in imbalance_metrics.items():
        report_lines.append(f"  {key}: {val}")
    report_lines.append("")

    model_readiness = compute_model_readiness(counts_df)
    report_lines.append("Model readiness recommendations:")
    for key, val in model_readiness.items():
        report_lines.append(f"  {key}: {val}")
    report_lines.append("")

    report_path = os.path.join(LOG_DIR, "eda_report.txt")
    save_report(report_lines, report_path)

    print("\n[EDA] Hoàn thành.", file=sys.stderr)
    print(f"  - Báo cáo lưu tại: {LOG_DIR}/")
    print("  - Xem eda_report.txt cùng các file CSV và PNG trong logs/eda/.")


if __name__ == "__main__":
    main()
