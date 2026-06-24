# Student Activity Classification

Hệ thống nhận diện trạng thái học tập sinh viên trong lớp học — **Group 3**.

5 class: `attentive` | `phone` | `laptop` | `sleeping` | `distracted`

---

## Cấu trúc project

```
student_activity/
├── README.md                 ← file này
├── requirements.txt
├── setup_project.py          ← tạo thư mục (chạy 1 lần)
├── config/
│   └── paths.py              ← đường dẫn tập trung
├── scripts/
│   ├── bootstrap.py          ← tự động (không chạy trực tiếp)
│   ├── data/                 ← pipeline dữ liệu
│   │   ├── extract_frames.py
│   │   ├── review_data.py
│   │   ├── check_dataset.py
│   │   ├── augment_data.py
│   │   ├── label_persons.py
│   │   └── split_by_person.py
│   └── model/                ← pipeline ML
│       ├── eda.py
│       ├── check_integrity.py
│       ├── train.py
│       └── evaluate.py
├── docs/
│   ├── data_collection.md    ← hướng dẫn quay video & extract
│   ├── modeling.md           ← hướng dẫn train & evaluate
│   ├── pipeline.md           ← sơ đồ tổng thể
│   ├── guides/               ← Colab / Kaggle
│   └── reports/              ← nhận xét kết quả
├── videos/                   ← video gốc (.mp4 / .mov)
├── dataset/
│   ├── raw/                  ← ảnh crop từ video
│   ├── augmented/            ← sau augmentation + nhãn người
│   ├── review/               ← grid review bằng mắt
│   └── split_by_person/      ← train / val / test
├── models/                   ← best_model.pth
└── logs/
    ├── eda/
    ├── training/
    └── evaluation/
```

> **Lưu ý:** Luôn chạy script từ **thư mục gốc** project:
> `python scripts/data/extract_frames.py`

---

## Quick Start

### 1. Cài đặt (1 lần)

```bash
pip install -r requirements.txt
python setup_project.py
```

### 2. Thu thập & xử lý dữ liệu

```bash
# Đặt video vào videos/, cấu hình VIDEO_MAP trong extract_frames.py
python scripts/data/extract_frames.py
python scripts/data/review_data.py       # xem dataset/review/, xóa ảnh lỗi
python scripts/data/check_dataset.py
python scripts/data/augment_data.py
python scripts/data/label_persons.py     # gán nhãn p1-p5 (K-Means toàn cục)
python scripts/data/split_by_person.py   # chia train/val/test theo người
```

### 3. Training & đánh giá

```bash
python scripts/model/eda.py              # tùy chọn — khảo sát dữ liệu
python scripts/model/check_integrity.py  # kiểm tra leakage
python scripts/model/train.py
python scripts/model/evaluate.py
```

---

## Chiến lược split theo người

| Split | Nội dung |
|---|---|
| **Train** | aug của p1, p2, p3, p4 |
| **Val** | raw của p1 + p5 (chọn model) |
| **Test** | raw của p1 + p5 (đánh giá cuối) |

- p1: đã thấy khi train — kiểm tra trên người quen  
- p5: hoàn toàn mới — kiểm tra khả năng tổng quát  
- Chống leakage: p1 chia theo `raw_idx`, không trùng aug↔raw giữa train và val/test

---

## Output chính

| Thành phần | Đường dẫn |
|---|---|
| Model | `models/best_model.pth` |
| Log EDA | `logs/eda/` |
| Log training | `logs/training/` |
| Log evaluation | `logs/evaluation/` |

---

## Tài liệu chi tiết

- [Thu thập dữ liệu](docs/data_collection.md)
- [Training & Evaluation](docs/modeling.md)
- [Pipeline tổng thể](docs/pipeline.md)
- [Colab / Kaggle](docs/guides/)

---

## 5 class — tóm tắt

| Class | Dấu hiệu chính |
|---|---|
| attentive | Nhìn về phía bảng / giảng viên |
| laptop | Mắt nhìn màn hình, tương tác laptop |
| phone | Cầm điện thoại, cúi nhìn |
| sleeping | Đầu gục, mắt nhắm |
| distracted | Quay hướng khác, không chú ý bài |

Ưu tiên khi overlap: **Sleeping > Phone > Laptop > Distracted > Attentive**
