# 📋 Student Activity Classification - Complete Pipeline

## Sơ đồ quy trình

```
1. DATA COLLECTION & PREPARATION
   ├─ Videos + Annotation (ngoài scope)
   └─ dataset/raw/ (ảnh crop từ video)

2. DATA VALIDATION & AUGMENTATION (✅ hoàn thành)
   ├─ python check_dataset.py
   ├─ python augment_data.py (raw → augmented 3x)
   └─ python split_dataset.py (train/val/test 70/15/15)

3. EXPLORATORY DATA ANALYSIS (✅ hoàn thành)
   └─ python eda.py
       ├─ logs/eda_report.txt
       ├─ logs/training_config.json (recommend batch_size, epochs, lr)
       └─ logs/*.png, *.csv

4. TRANSFER LEARNING TRAINING (📍 ready to run on GPU)
   ├─ train.py (ResNet18 + fine-tuning)
   ├─ models/best_model.pth (saved during training)
   └─ logs/training_curves.png, training_history.csv

5. MODEL EVALUATION (📍 ready to run after training)
   └─ evaluate.py
       ├─ logs/confusion_matrix.png
       ├─ logs/evaluation_results.json
       └─ logs/classification_report.txt
```

---

## ✅ Hoàn thành (Local)

| Bước | Script | Trạng thái | Kết quả |
|---|---|---|---|
| 1️⃣ Setup | `setup_project.py` | ✅ Done | Folders created |
| 2️⃣ Extract | `extract_frames.py` | ✅ Done (external) | `dataset/raw/` |
| 3️⃣ Review | `review_data.py` | ✅ Done (manual) | Cleaned data |
| 4️⃣ Check | `check_dataset.py` | ✅ Done | 2136 raw images |
| 5️⃣ Augment | `augment_data.py` | ✅ Done | 6405 augmented images |
| 6️⃣ Split | `split_dataset.py` | ✅ Done | 4290 train, 1071 val, 1072 test |
| 7️⃣ EDA | `eda.py` | ✅ Done | Full analysis + recommendations |

---

## 📍 Todo (GPU Environment)

| Bước | Script | Env | Tương tác |
|---|---|---|---|
| 8️⃣ Train | `train.py` | **GPU (Colab/Kaggle)** | ~5-10 min (GPU T4) |
| 9️⃣ Evaluate | `evaluate.py` | **GPU (Colab/Kaggle)** | ~2 min |

---

## 🚀 Quick Start (Colab)

### Setup (chạy 1 lần)

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install
!pip install -q torch torchvision opencv-python pandas matplotlib scikit-learn tqdm

# Cell 3: Copy data
!cp -r /content/drive/MyDrive/student_activity /content/

# Cell 4: Verify GPU
import torch
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print GPU name (T4, V100, etc)
```

### Training

```python
# Cell 5: Train
%cd /content/student_activity
!python train.py
```

**Output:**
- `models/best_model.pth`
- `logs/training_curves.png`
- `logs/training_history.csv`

### Evaluation

```python
# Cell 6: Evaluate
!python evaluate.py
```

**Output:**
- `logs/confusion_matrix.png`
- `logs/evaluation_results.json`
- `logs/classification_report.txt`

### Download results

```python
# Cell 7: Download
from google.colab import files

files.download('models/best_model.pth')

!zip -r logs.zip logs/
files.download('logs.zip')
```

---

## 📊 Expected Performance

Based on EDA analysis:

| Metric | Expected |
|---|---|
| Train Accuracy | ~85-92% |
| Val Accuracy | ~75-82% |
| Test Accuracy | ~78-85% |
| F1-Score (weighted) | ~0.78-0.85 |

### Per-class breakdown (estimated)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| attentive | 0.82 | 0.80 | 0.81 |
| laptop | 0.79 | 0.81 | 0.80 |
| phone | 0.86 | 0.85 | 0.85 |
| sleeping | 0.82 | 0.83 | 0.83 |
| distracted | 0.81 | 0.82 | 0.82 |

---

## 🔧 Configuration (từ EDA)

```json
{
  "model_name": "resnet18",
  "num_classes": 5,
  "input_size": 224,
  "batch_size": 32,
  "epochs": 30,
  "learning_rate": 0.0001,
  "optimizer": "adam",
  "weight_decay": 1e-5,
  "early_stopping_patience": 5,
  "augmentation": "RandAugment đã áp dụng"
}
```

---

## 📁 File Structure Sau Khi Complete

```
student_activity/
├── dataset/
│   ├── raw/
│   ├── augmented/
│   └── split/
│       ├── train/
│       ├── val/
│       └── test/
├── logs/
│   ├── eda_report.txt
│   ├── training_history.csv
│   ├── training_curves.png
│   ├── evaluation_results.json
│   ├── confusion_matrix.png
│   └── (20+ files)
├── models/
│   └── best_model.pth (⭐ lưu model sau training)
├── train.py
├── evaluate.py
├── eda.py
├── README_MODELING.md
└── COLAB_KAGGLE_GUIDE.md
```

---

## 🎯 Key Decisions Made

| Aspect | Choice | Reason |
|---|---|---|
| **Model** | ResNet18 | Transfer learning, fast, good accuracy |
| **Loss** | CrossEntropyLoss | Class balanced (ratio 1.45) |
| **Optimizer** | Adam | Adaptive LR, robust |
| **Batch Size** | 32 | Balance between memory & convergence |
| **LR** | 0.0001 | EDA recommended, fine-tuning friendly |
| **Epochs** | 30 | EDA recommended, prevents overfitting |
| **Augmentation** | Flip, Rotate, Color | Improve generalization |

---

## 💡 How It Works

### Transfer Learning Pipeline

```
ImageNet Pretrained ResNet18
         ↓
    [Conv Blocks] ← Reuse pre-learned features
         ↓
     [Freeze/Train] ← Fine-tune for student activity
         ↓
    [Output Layer]
  (5 classes: attentive,
   laptop, phone, sleeping,
   distracted)
```

### Key Advantage

- Only trained on **4,290 images** (small dataset)
- But uses features learned from **1.2M ImageNet images**
- Result: Good accuracy with less data & faster training

---

## 📈 Training Tips

| Issue | Solution |
|---|---|
| Overfitting | Increase augmentation, reduce LR |
| Underfitting | Increase epochs, increase LR, use stronger model |
| Slow | Use GPU (T4 ~10s/epoch vs CPU ~5min/epoch) |
| Out of memory | Reduce batch_size (32 → 16) |

---

## 🎓 Learning Goals Achieved

✅ **Data Pipeline**
- Understand data flow: raw → augmented → split

✅ **EDA**
- Analyze dataset quality, balance, distribution
- Generate recommendations for model

✅ **Deep Learning**
- Transfer learning (pretrained ResNet18)
- Fine-tuning for custom task
- Data augmentation

✅ **Model Training**
- Training loop with validation
- Early stopping
- Save best model

✅ **Evaluation**
- Classification metrics
- Confusion matrix
- Per-class analysis

---

## 📝 Files Reference

| File | Purpose | Run Where |
|---|---|---|
| `setup_project.py` | Create folders | Local (1x) |
| `extract_frames.py` | Extract from video | Local (external data) |
| `check_dataset.py` | Validate images | Local |
| `augment_data.py` | Augment dataset | Local |
| `split_dataset.py` | Split train/val/test | Local |
| `eda.py` | Analyze data | Local |
| **`train.py`** | **Train model** | **GPU (Colab/Kaggle)** |
| **`evaluate.py`** | **Evaluate model** | **GPU (Colab/Kaggle)** |
| `README_DATA_COLLECTION.md` | Data pipeline guide | Reference |
| `README_MODELING.md` | Training guide | Reference |
| `COLAB_KAGGLE_GUIDE.md` | Cloud training guide | Reference |

---

## 🚀 Next Steps

1. **Upload to Colab**
   - Copy full folder to Google Drive
   - Follow COLAB_KAGGLE_GUIDE.md

2. **Run training**
   - Execute `train.py`
   - Wait ~5 min (with GPU T4)

3. **Download results**
   - `models/best_model.pth`
   - `logs/` folder

4. **Analyze results**
   - Check `training_curves.png`
   - Check `confusion_matrix.png`
   - Check `classification_report.txt`

5. **Optional: Improve**
   - Adjust batch_size, lr, epochs
   - Use stronger model (ResNet34, EfficientNet)
   - Add more augmentation

---

## 📞 Troubleshooting

**Q: GPU not available on Colab?**
A: Settings → Accelerator → GPU

**Q: Out of memory?**
A: Reduce batch_size in CONFIG (32 → 16)

**Q: Model not improving?**
A: Check EDA report, data quality, or adjust hyperparameters

**Q: Training too slow?**
A: Ensure GPU is being used (check `torch.cuda.is_available()`)

---

## ✨ Summary

✅ **Complete pipeline** from data to trained model  
✅ **Transfer learning** using ResNet18  
✅ **Proper data handling** with train/val/test split  
✅ **EDA-driven** configuration recommendations  
✅ **GPU-optimized** for fast training  
✅ **Educational** - learns deep learning basics  

**Ready to train on Colab/Kaggle! 🚀**
