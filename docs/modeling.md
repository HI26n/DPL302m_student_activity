# Hướng dẫn Training & Evaluation

> Xem [README.md](../README.md) để biết cấu trúc project đầy đủ.

## Pipeline modeling

```bash
python scripts/model/eda.py              # khảo sát dữ liệu (tùy chọn)
python scripts/model/check_integrity.py  # kiểm tra leakage
python scripts/model/train.py            # ResNet18 transfer learning
python scripts/model/evaluate.py         # đánh giá test set
```

## Scripts trong `scripts/model/`

| File | Mô tả |
|---|---|
| `eda.py` | Phân tích raw/augmented/split, xuất `logs/eda/` |
| `check_integrity.py` | Kiểm tra leakage trên `split_by_person` |
| `train.py` | Train ResNet18, lưu `models/best_model.pth` |
| `evaluate.py` | Accuracy, F1, confusion matrix → `logs/evaluation/` |

## Cấu hình training

| Tham số | Giá trị |
|---|---|
| Model | ResNet18 (ImageNet pretrained) |
| Input | 224×224 |
| Optimizer | Adam, lr=0.0001 |
| Batch size | 32 |
| Epochs | 30 (early stopping patience=5) |
| Data | `dataset/split_by_person/` |

### Augmentation khi train

`RandomHorizontalFlip`, `RandomRotation(10°)`, `ColorJitter`

Val/test: chỉ resize + normalize (không augment).

## Output

| File | Mô tả |
|---|---|
| `models/best_model.pth` | Model tốt nhất theo val accuracy |
| `logs/training/training_history.csv` | Loss/acc từng epoch |
| `logs/training/training_curves.png` | Biểu đồ train/val |
| `logs/training/training_config.json` | Cấu hình đã dùng |
| `logs/training/final_test_results.json` | Test acc sau training |
| `logs/evaluation/evaluation_results.json` | Kết quả chi tiết |
| `logs/evaluation/confusion_matrix.png` | Ma trận nhầm lẫn |

## Đánh giá kết quả

`evaluate.py` in và lưu:

- Overall Accuracy & Weighted F1
- Precision / Recall / F1 từng class
- Confusion matrix

Test set gồm **p1** (đã thấy khi train) và **p5** (người mới).

## Lưu ý

- Val acc cao bất thường ngay epoch 1 → kiểm tra leakage bằng `check_integrity.py`
- Class `sleeping` thường ít mẫu test → kết quả có thể không đại diện
- Train trên GPU (Colab/Kaggle) nhanh hơn — xem `docs/guides/`

## Chạy trên Colab / Kaggle

Xem [guides/COLAB_TRAINING_GUIDE.md](guides/COLAB_TRAINING_GUIDE.md) và [guides/COLAB_KAGGLE_GUIDE.md](guides/COLAB_KAGGLE_GUIDE.md).

Upload project, cài `requirements.txt`, chạy cùng các lệnh `python scripts/...` từ thư mục gốc.
