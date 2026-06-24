# Hướng dẫn Thu thập & Xử lý Dữ liệu

> Xem [README.md](../README.md) để biết cấu trúc project đầy đủ.

## Pipeline dữ liệu

```
videos/  →  extract  →  review  →  check  →  augment  →  label  →  split
              ↓            ↓                    ↓           ↓          ↓
         dataset/raw   dataset/review    dataset/augmented   p1-p5   split_by_person/
```

## Thứ tự chạy

```bash
python setup_project.py                    # 1 lần

python scripts/data/extract_frames.py      # YOLOv8 crop từ video
python scripts/data/review_data.py         # grid review bằng mắt
python scripts/data/check_dataset.py       # kiểm tra số lượng (≥150/class)
python scripts/data/augment_data.py        # tăng cường ×3
python scripts/data/label_persons.py       # gán nhãn người p1-p5
python scripts/data/split_by_person.py     # chia train/val/test
```

## Scripts trong `scripts/data/`

| File | Mô tả |
|---|---|
| `extract_frames.py` | YOLOv8 detect người, crop, lọc chất lượng → `dataset/raw/` |
| `review_data.py` | Tạo grid `dataset/review/review_<class>.jpg` |
| `check_dataset.py` | Đếm ảnh, kiểm tra 224×224 |
| `augment_data.py` | Augment → `dataset/augmented/` |
| `label_persons.py` | K-Means toàn cục, gán `p1_`…`p5_` |
| `split_by_person.py` | Chia `dataset/split_by_person/` theo người |

## Cấu hình extract

1. Đặt video vào `videos/`
2. Mở `scripts/data/extract_frames.py` → chỉnh `VIDEO_MAP`
3. Đặt tên video: `[class]_[số].mp4` hoặc `.mov`

```python
VIDEO_MAP = [
    ("videos/attentive_1.mp4", "dataset/raw/attentive"),
    ("videos/phone_1.mp4",     "dataset/raw/phone"),
    # ...
]
```

`CLEAN_BEFORE_EXTRACT = True` — xóa ảnh cũ trước khi extract lại.

## Định nghĩa 5 class

| Class | Định nghĩa |
|---|---|
| **attentive** | Đầu/mắt hướng bảng hoặc GV |
| **laptop** | Mắt nhìn màn hình, tay tương tác laptop |
| **phone** | Tay cầm điện thoại rõ ràng |
| **sleeping** | Đầu gục, mắt nhắm |
| **distracted** | Quay hướng khác, không chú ý |

Laptop mở trên bàn là bình thường ở mọi class — classifier dựa vào **hành vi**, không phải sự có mặt của laptop.

Ưu tiên overlap: `Sleeping > Phone > Laptop > Distracted > Attentive`

## Review thủ công

Sau `review_data.py`, mở `dataset/review/` và xóa ảnh lỗi trong `dataset/raw/<class>/`:

- Chỉ thấy mặt, không thấy tay/đồ vật
- Ảnh mờ, bị che, trạng thái chuyển tiếp
- Laptop class: người nhìn bảng → nên là attentive

## Xử lý lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install -r requirements.txt` |
| Không tìm thấy video | Kiểm tra tên file trong `videos/` và `VIDEO_MAP` |
| Tỉ lệ giữ lại < 50% | Camera quá xa, quay lại gần hơn |
| p1 khác nhau giữa class | Chạy lại `label_persons.py` (K-Means toàn cục) |
