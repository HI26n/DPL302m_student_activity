# Hướng dẫn Training & Evaluation

## Tổng quan

Phần này sử dụng **Transfer Learning** với ResNet18 từ ImageNet pretrained weights, giúp bạn dễ dàng học cách xây dựng deep learning pipeline mà không cần xây từ đầu.

### Các file

| File | Mô tả |
|---|---|
| `train.py` | Training script dùng ResNet18 |
| `evaluate.py` | Đánh giá model trên test set |

---

## Cấu trúc Training

```python
1. Load dữ liệu từ dataset/split/train, split/val
2. ResNet18 pretrained + fine-tune output layer
3. Augmentation cơ bản (flip, rotate, color jitter)
4. Early stopping nếu val acc không cải thiện
5. Lưu best model vào models/best_model.pth
```

---

## Chạy Training

### Bước 1: Đảm bảo dữ liệu đã sẵn sàng

```bash
# Kiểm tra dữ liệu
ls -la dataset/split/train/
ls -la dataset/split/val/
```

### Bước 2: Chạy training

```bash
python train.py
```

**Output:**
- Sẽ in log mỗi epoch với train/val loss và accuracy
- Lưu best model: `models/best_model.pth`
- Training curves: `logs/training_curves.png`
- Training history: `logs/training_history.csv`
- Config: `logs/training_config.json`

**Ví dụ output:**

```
[Epoch 1/30]
Train Loss: 1.5432 | Train Acc: 45.23% | Val Loss: 1.3421 | Val Acc: 52.10%
  ✅ Best model saved: models/best_model.pth (Acc: 52.10%)

[Epoch 2/30]
Train Loss: 1.2134 | Train Acc: 58.45% | Val Loss: 1.1234 | Val Acc: 61.34%
  ✅ Best model saved: models/best_model.pth (Acc: 61.34%)
...
```

---

## Evaluate Model

Sau khi training hoàn tất, evaluate trên test set:

```bash
python evaluate.py
```

**Output:**
- Overall Accuracy
- Weighted F1-Score
- Classification Report (Precision, Recall, F1 từng class)
- Confusion Matrix (hình)
- Results: `logs/evaluation_results.json`

**Ví dụ output:**

```
RESULTS
============================================================
Overall Accuracy: 0.8234
Weighted F1-Score: 0.8156

CLASSIFICATION REPORT
============================================================
           precision    recall  f1-score   support
  attentive       0.85      0.82      0.84       207
     laptop       0.79      0.81      0.80       256
     phone        0.88      0.86      0.87       207
   sleeping       0.82      0.84      0.83       176
 distracted       0.81      0.83      0.82       226
      micro avg   0.82      0.83      0.82      1072
      macro avg   0.83      0.83      0.83      1072
 weighted avg     0.82      0.83      0.82      1072
```

---

## Transfer Learning - Làm sao nó hoạt động?

### Ý tưởng chính

1. **Pretrained weights**: ResNet18 được train trên ImageNet (1.2M ảnh, 1000 classes)
2. **Feature extraction**: Layer đầu tiên (đã train) đã học cách detect cạnh, texture, shape
3. **Fine-tuning**: Chỉ thay lớp cuối cùng (fc layer) từ 1000 classes → 5 classes
4. **Lợi ích**: Cần ít dữ liệu hơn, train nhanh hơn

### Hình minh họa

```
ImageNet pretrained:
Input → Conv1 → Conv2 → Conv3 → Conv4 → Conv5 → fc(1000) → Output
        (chuyên detect features)                    ↓
                                                    Thay này

Student Activity fine-tuned:
Input → Conv1 → Conv2 → Conv3 → Conv4 → Conv5 → fc(5) → Output
        (sử dụng lại)                              ↑
                                                    Thay bằng này
```

---

## Các tham số quan trọng

Trong `train.py`, bạn có thể thay đổi CONFIG:

```python
CONFIG = {
    "model_name": "resnet18",      # hoặc "resnet34", "efficientnet_b0"
    "num_classes": 5,
    "batch_size": 32,              # tăng lên 64 nếu GPU có mem đủ
    "epochs": 30,                  # số lần qua hết training data
    "learning_rate": 0.0001,       # tốc độ học
    "early_stopping_patience": 5,  # stop nếu val acc không cải thiện trong 5 epochs
}
```

### Gợi ý điều chỉnh

| Vấn đề | Giải pháp |
|---|---|
| Overfitting (train acc cao nhưng val acc thấp) | Tăng augmentation, tăng weight_decay |
| Underfitting (cả train/val acc thấp) | Tăng epochs, tăng learning rate |
| Train quá chậm | Tăng batch_size, giảm số epochs |
| GPU memory not enough | Giảm batch_size, dùng model nhẹ hơn (efficientnet_b0) |

---

## Các Models có sẵn

Bạn có thể thay `model_name` trong CONFIG:

| Model | Tốc độ | Độ chính xác | Memory |
|---|---|---|---|
| **resnet18** | Nhanh | Tốt | Thấp |
| resnet34 | Bình thường | Tốt hơn | Bình thường |
| efficientnet_b0 | Chuẩn | Rất tốt | Bình thường |

**Khuyến nghị**: Dùng **resnet18** cho lần đầu vì nhanh và dễ debug.

---

## Augmentation

Script đã áp dụng các augmentation sau cho training:

```python
- RandomHorizontalFlip (lật ảnh ngang)
- RandomRotation (xoay 10 độ)
- ColorJitter (thay đổi brightness, contrast)
- Normalize (chuẩn hóa với ImageNet mean/std)
```

Nếu muốn thêm augmentation khác, sửa trong `train.py`:

```python
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # Thêm vào đây, ví dụ:
    # transforms.GaussianBlur(kernel_size=3),
    # transforms.RandomAffine(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    ...
])
```

---

## Log và Output

Sau khi chạy train + evaluate, bạn sẽ có:

```
logs/
├── training_history.csv          ← DataFrame với train/val loss/acc từng epoch
├── training_curves.png           ← Graph loss/acc
├── training_config.json          ← Config đã dùng
├── evaluation_results.json       ← Accuracy, F1-score, classification report
├── classification_report.txt     ← Dạng text của report
├── confusion_matrix.png          ← Heatmap
├── eda_report.txt
├── eda_*.csv
└── eda_*.png

models/
└── best_model.pth                ← Best model được lưu

dataset/
├── raw/
├── augmented/
└── split/
    ├── train/
    ├── val/
    └── test/
```

---

## Troubleshooting

### Error: "CUDA out of memory"
→ Giảm batch_size từ 32 xuống 16 hoặc 8

### Error: "No images found"
→ Kiểm tra dữ liệu đã được split chưa: `python split_dataset.py`

### Validation accuracy không tăng
→ Tăng epochs, giảm learning rate, hoặc kiểm tra dữ liệu có bị lỗi không

### Training quá chậm
→ Tăng batch_size (nếu GPU mem đủ), dùng GPU nếu có

---

## Next Steps

1. ✅ Chạy `train.py` → nhận được best model
2. ✅ Chạy `evaluate.py` → nhận kết quả trên test set
3. 🔄 Điều chỉnh tham số → thử lại
4. 📊 Xem training curves → kiểm tra overfitting/underfitting
5. 🚀 Deploy model nếu accuracy đủ cao

---

## Reference

- [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ImageNet Pretrained Models](https://pytorch.org/vision/stable/models.html)
