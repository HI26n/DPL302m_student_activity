# 🚀 Hướng dẫn Train trên Google Colab với Dataset từ Google Drive

---

## 📋 **Bước 1: Chuẩn bị Dataset (Máy Local)**

### **1.1 Chắc chắn dataset đã được split**
```powershell
cd d:\minhcuong\FPTU\Term_5\DPL302m\student_activity
python split_dataset.py
```

Sau đó sẽ có folder cấu trúc:
```
dataset/split/
├── train/
│   ├── attentive/ (ảnh)
│   ├── phone/ (ảnh)
│   ├── laptop/ (ảnh)
│   ├── sleeping/ (ảnh)
│   └── distracted/ (ảnh)
└── val/
    ├── attentive/ (ảnh)
    ├── phone/ (ảnh)
    ├── laptop/ (ảnh)
    ├── sleeping/ (ảnh)
    └── distracted/ (ảnh)
```

### **1.2 Compress dataset thành ZIP**
```powershell
# Windows - Sử dụng 7-Zip hoặc Windows Explorer
# Chuột phải trên folder dataset/split → Compress (hoặc Send to → Compressed folder)
# Hoặc dùng PowerShell:

Compress-Archive -Path "dataset/split" -DestinationPath "student_activity_dataset.zip" -Force
```

**Kích thước file:** ~100-200 MB (tùy thuộc số ảnh)

---

## ☁️ **Bước 2: Upload lên Google Drive**

1. Vào https://drive.google.com
2. Tạo **New Folder** → Đặt tên: `DPL302m_Data`
3. **Upload** file `student_activity_dataset.zip` vào folder đó
4. **Chuột phải** trên file → **Shareable Link** → Copy link

   **Ví dụ link:**
   ```
   https://drive.google.com/file/d/1ABC_xyz123/view?usp=sharing
   ```

---

## 🔗 **Bước 3: Tạo Google Colab Notebook**

### **3.1 Tạo notebook mới**
1. Vào https://colab.research.google.com
2. Click **File** → **New notebook** → Đặt tên: `DPL302m_Training`
3. Click **Files icon** (⬅️) → **Connect to Google Drive**

### **3.2 Mount Google Drive vào Colab**

**Cell 1: Mount Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted!")
```

Run: `Shift + Enter`

---

## 📥 **Bước 4: Download & Extract Dataset**

**Cell 2: Download từ Drive**
```python
import os
import shutil

# Tạo thư mục working
os.makedirs('/content/working', exist_ok=True)
os.chdir('/content/working')

# Copy dataset từ Google Drive
source = '/content/drive/MyDrive/DPL302m_Data/student_activity_dataset.zip'
dest = '/content/working/student_activity_dataset.zip'

if os.path.exists(source):
    shutil.copy(source, dest)
    print(f"✅ Copied: {dest}")
else:
    print(f"❌ File not found: {source}")
    print("Please check your Google Drive path!")
```

Run: `Shift + Enter`

**Cell 3: Extract Dataset**
```python
import zipfile

# Extract zip
with zipfile.ZipFile('/content/working/student_activity_dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/working')

# Kiểm tra cấu trúc
import os
print("📁 Extracted files:")
os.system('ls -la /content/working/')

# Check training data
train_path = '/content/working/split/train'
if os.path.exists(train_path):
    for cls in os.listdir(train_path):
        cls_path = os.path.join(train_path, cls)
        count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"  {cls}: {count} images")
else:
    print(f"❌ Train folder not found at {train_path}")
```

Run: `Shift + Enter`

---

## 📦 **Bước 5: Clone Repo & Install Dependencies**

**Cell 4: Clone GitHub Repo**
```python
!git clone https://github.com/HI26n/DPL302m_student_activity.git
%cd DPL302m_student_activity
!pip install -r requirements.txt -q
print("✅ Repo cloned and dependencies installed!")
```

Run: `Shift + Enter`

---

## 🎯 **Bước 6: Chuẩn bị Dataset cho Training**

**Cell 5: Link Dataset vào Colab**
```python
import os
import shutil

# Copy dataset vào Colab repo
train_src = '/content/working/split/train'
val_src = '/content/working/split/val'

train_dst = '/content/DPL302m_student_activity/dataset/split/train'
val_dst = '/content/DPL302m_student_activity/dataset/split/val'

os.makedirs('/content/DPL302m_student_activity/dataset/split', exist_ok=True)

# Copy nếu chưa có
if not os.path.exists(train_dst):
    print("Copying training data...")
    shutil.copytree(train_src, train_dst)
    print("✅ Training data copied")

if not os.path.exists(val_dst):
    print("Copying validation data...")
    shutil.copytree(val_src, val_dst)
    print("✅ Validation data copied")

# Verify
print("\n📊 Dataset verification:")
print(f"Train images: {sum([len(f) for r, d, f in os.walk(train_dst)])}")
print(f"Val images: {sum([len(f) for r, d, f in os.walk(val_dst)])}")
```

Run: `Shift + Enter`

---

## 🚀 **Bước 7: CHẠY TRAINING**

**Cell 6: Chạy Training**
```python
%cd /content/DPL302m_student_activity

from train_kaggle import train_kaggle_model

print("🔥 Starting training on GPU...\n")
results = train_kaggle_model(resume_from_checkpoint=False)

print(f"\n✅ TRAINING COMPLETED!")
print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
print(f"Model saved at: {results['model_path']}")
```

Run: `Shift + Enter` (Chờ training hoàn thành - ~2 giờ)

---

## 📊 **Bước 8 (Optional): Xem Kết quả**

**Cell 7: Visualize Training Results**
```python
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Load training history
history = pd.read_csv('/content/DPL302m_student_activity/logs/training_history.csv')
print("Training History:")
print(history.tail(10))

# Plot training curves
img = Image.open('/content/DPL302m_student_activity/logs/training_curves.png')
plt.figure(figsize=(14, 5))
plt.imshow(img)
plt.axis('off')
plt.title('Training Curves')
plt.tight_layout()
plt.show()

print("\n✅ Training completed successfully!")
```

Run: `Shift + Enter`

---

## 💾 **Bước 9: Download Model**

**Cell 8: Download Results**
```python
from google.colab import files

# Download model
files.download('/content/DPL302m_student_activity/models/best_model.pth')

# Download logs
!zip -r /content/logs.zip /content/DPL302m_student_activity/logs/
files.download('/content/logs.zip')

print("✅ Files downloaded to your computer!")
```

Run: `Shift + Enter`

---

## 🔄 **Alternative: Chạy toàn bộ một lần**

Nếu muốn chạy cả quá trình một lần mà không cần tách từng cell:

**Notebook Complete Code:**
```python
# ===== CELL 1: Mount Drive =====
from google.colab import drive
drive.mount('/content/drive')

# ===== CELL 2: Setup =====
import os
import shutil
import zipfile

os.makedirs('/content/working', exist_ok=True)
os.chdir('/content/working')

# Copy từ Drive
source = '/content/drive/MyDrive/DPL302m_Data/student_activity_dataset.zip'
dest = '/content/working/student_activity_dataset.zip'
shutil.copy(source, dest)

# Extract
with zipfile.ZipFile(dest, 'r') as zip_ref:
    zip_ref.extractall('/content/working')

print("✅ Dataset extracted!")

# ===== CELL 3: Clone & Install =====
!git clone https://github.com/HI26n/DPL302m_student_activity.git
%cd DPL302m_student_activity
!pip install -r requirements.txt -q

# ===== CELL 4: Copy Dataset =====
import shutil
train_src = '/content/working/split/train'
val_src = '/content/working/split/val'
train_dst = '/content/DPL302m_student_activity/dataset/split/train'
val_dst = '/content/DPL302m_student_activity/dataset/split/val'

os.makedirs('/content/DPL302m_student_activity/dataset/split', exist_ok=True)
shutil.copytree(train_src, train_dst)
shutil.copytree(val_src, val_dst)

print("✅ Dataset ready!")

# ===== CELL 5: TRAIN =====
from train_kaggle import train_kaggle_model
results = train_kaggle_model()
print(f"\n✅ Best Accuracy: {results['best_val_acc']:.2f}%")

# ===== CELL 6: Download =====
from google.colab import files
files.download('/content/DPL302m_student_activity/models/best_model.pth')
print("✅ Model downloaded!")
```

---

## ✅ **Checklist**

- [ ] Dataset compressed thành ZIP
- [ ] ZIP uploaded lên Google Drive
- [ ] Google Drive link sẵn
- [ ] Colab notebook tạo
- [ ] GPU enabled (Runtime → Change runtime type → GPU)
- [ ] Chạy từng cell theo thứ tự

---

## 🎬 **Quick TL;DR**

1. **Local**: `Compress-Archive -Path "dataset/split" -DestinationPath "dataset.zip"`
2. **Drive**: Upload `dataset.zip` → Copy link
3. **Colab**: 
   - Mount Drive
   - Extract dataset
   - Clone repo
   - `train_kaggle_model()`
   - Download model

**Total time:** ~2.5 giờ (vs 30 giờ trên CPU local) ⚡

---

## 🆘 **Troubleshoot**

| Lỗi | Giải pháp |
|-----|----------|
| `FileNotFoundError: dataset` | Kiểm tra path trong Cell, phải là `/content/working/split/` |
| `No GPU found` | Settings → Runtime → Change runtime type → GPU |
| `Out of Memory` | Giảm batch_size trong config |
| `Permission denied` | Kiểm tra Drive sharing link public |

---

**Bạn ready chưa?** 🚀
