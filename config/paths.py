"""
Đường dẫn tập trung cho toàn project.
Mọi script chạy từ thư mục gốc: python scripts/data/extract_frames.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Thư mục gốc ──────────────────────────────────────────────
VIDEOS_DIR = PROJECT_ROOT / "videos"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

# ── Dataset ───────────────────────────────────────────────────
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_RAW = DATASET_DIR / "raw"
DATASET_AUGMENTED = DATASET_DIR / "augmented"
DATASET_REVIEW = DATASET_DIR / "review"
DATASET_SPLIT = DATASET_DIR / "split_by_person"

SPLIT_TRAIN = DATASET_SPLIT / "train"
SPLIT_VAL = DATASET_SPLIT / "val"
SPLIT_TEST = DATASET_SPLIT / "test"

# ── Logs ──────────────────────────────────────────────────────
LOGS_EDA = LOGS_DIR / "eda"
LOGS_TRAINING = LOGS_DIR / "training"
LOGS_EVALUATION = LOGS_DIR / "evaluation"

# ── Model checkpoint ──────────────────────────────────────────
BEST_MODEL = MODELS_DIR / "best_model.pth"
YOLO_WEIGHTS = PROJECT_ROOT / "yolov8n.pt"

# ── Constants ───────────────────────────────────────────────
CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
AUGMENT_FACTOR = 2
NUM_PERSONS = 5


def p(path: Path) -> str:
    """Chuyển Path → str cho os.path / OpenCV."""
    return str(path)


def class_dir(base: Path, cls_name: str) -> Path:
    return base / cls_name


def setup_sys_path():
    """Thêm project root vào sys.path (gọi ở đầu mỗi script)."""
    import sys
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
