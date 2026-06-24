# Chạy Training trên Google Colab / Kaggle

## Google Colab (khuyến nghị)

### Bước 1: Upload dữ liệu lên Colab

**Cách 1: Upload từ Google Drive (dễ nhất)**

```python
# Chạy cell này trong Colab
from google.colab import drive
drive.mount('/content/drive')
```

Sau đó upload thư mục `student_activity` lên Google Drive (khoảng 500MB-1GB)

**Cách 2: Upload từ file (nếu < 2GB)**

```python
# Click menu Upload → chọn file .zip của project
# Sau đó unzip
!unzip student_activity.zip
%cd student_activity
```

### Bước 2: Cài đặt dependencies

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q opencv-python pandas matplotlib scikit-learn tqdm albumentations
```

### Bước 3: Chạy training

```python
# Chạy trên GPU (Colab free có T4 GPU)
!python train.py
```

**Lưu ý**: Colab sẽ tự enable GPU nếu chạy model training. Kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # Sẽ in True nếu có GPU
print(torch.cuda.get_device_name(0))  # Sẽ in tên GPU
```

---

## Kaggle Notebook

### Bước 1: Upload dataset

1. Vào https://www.kaggle.com/datasets/create
2. Upload thư mục `dataset/` (hoặc toàn bộ project)
3. Note lại dataset ID (ví dụ: `your_username/student_activity_dataset`)

### Bước 2: Tạo Kaggle Notebook

```python
# Cell 1: Setup
!pip install -q torch torchvision torchaudio
!pip install -q opencv-python pandas matplotlib scikit-learn tqdm

# Cell 2: Copy dataset
!cp -r /kaggle/input/student_activity_dataset/student_activity /tmp/
%cd /tmp/student_activity

# Cell 3: Chạy training
!python train.py
```

### Bước 3: Enable GPU

- Settings → Accelerator → GPU (P100 hoặc T4)

---

## Tối ưu cho GPU - Quick Start Script

Nếu muốn training nhanh hơn, dùng script này (chạy với GPU T4 ~1-2h cho 30 epochs):

```python
# train_gpu.py - tối ưu cho GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # dùng GPU 0

import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Import từ train.py
import sys
sys.path.append('.')
from train import main, CONFIG

# Điều chỉnh config cho GPU
CONFIG["batch_size"] = 64  # lớn hơn vì GPU có mem
CONFIG["epochs"] = 30

# Chạy
main()
```

---

## So sánh Tốc độ

| Device | 1 Epoch | 30 Epochs | Ghi chú |
|---|---|---|---|
| **CPU (local)** | ~5 phút | ~150 phút (2.5h) | Quá chậm |
| **GPU T4 (Colab)** | ~10 giây | ~5 phút | Nhanh! |
| **GPU P100 (Kaggle)** | ~5 giây | ~2.5 phút | Rất nhanh! |

---

## Colab Cell-by-Cell Template

Copy paste các cell này vào Colab notebook:

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Cài thư viện
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q opencv-python pandas matplotlib scikit-learn tqdm
```

### Cell 3: Copy project từ Drive (nếu upload từ Drive)
```python
import shutil
shutil.copytree('/content/drive/MyDrive/student_activity', '/content/student_activity')
%cd /content/student_activity
```

### Cell 4: Check GPU
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Cell 5: Run Training
```python
!python train.py
```

### Cell 6: Evaluate
```python
!python evaluate.py
```

### Cell 7: Download results
```python
from google.colab import files

# Download best model
files.download('models/best_model.pth')

# Download logs
!zip -r logs.zip logs/
files.download('logs.zip')
```

---

## Kaggle Cell-by-Cell Template

### Cell 1: Setup
```python
!pip install -q torch torchvision opencv-python
%cd /kaggle/working
```

### Cell 2: Prepare data
```python
import shutil
shutil.copytree('/kaggle/input/student-activity-data/student_activity', 
                 '/kaggle/working/student_activity')
%cd /kaggle/working/student_activity
```

### Cell 3: Check GPU
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Cell 4: Run
```python
!python train.py
```

---

## Lưu kết quả sau training

**Trên Colab:**
```python
from google.colab import files

# Download model
files.download('models/best_model.pth')

# Download logs
!tar -czf logs.tar.gz logs/
files.download('logs.tar.gz')
```

**Trên Kaggle:**
```python
# Kết quả sẽ tự lưu trong `/kaggle/working`
# Xem thẳng trong notebook hoặc download từ output panel
```

---

## Troubleshooting

### "CUDA out of memory" trên GPU
→ Giảm batch_size từ 64 xuống 32 hoặc 16

```python
CONFIG["batch_size"] = 32
```

### GPU không được detect
→ Kiểm tra
```python
import torch
print(torch.cuda.is_available())
```

Nếu `False`, vào Settings → GPU

### Download file bị timeout
→ Dùng `!zip -r` để nén trước

```python
!zip -r logs.zip logs/
files.download('logs.zip')
```

---

## Khuyến nghị

1. **Đầu tiên**: Dùng **Google Colab** (free, dễ, GPU T4 đủ nhanh)
2. **Nếu cần nhanh hơn**: Dùng **Kaggle** (GPU P100 nhanh hơn)
3. **Output**: Lưu `best_model.pth` + `logs/` để dùng sau

---

## Dự kiến kết quả (GPU T4)

```
[Epoch 1/30]
Train Loss: 1.5432 | Train Acc: 45.23% | Val Loss: 1.3421 | Val Acc: 52.10%
  ✅ Best model saved (Acc: 52.10%) - 10 giây

[Epoch 2/30]
Train Loss: 1.2134 | Train Acc: 58.45% | Val Loss: 1.1234 | Val Acc: 61.34%
  ✅ Best model saved (Acc: 61.34%) - 10 giây

... (25 epochs tiếp theo) ...

[Epoch 30/30]
Train Loss: 0.1234 | Train Acc: 95.23% | Val Loss: 0.5123 | Val Acc: 82.45%

TRAINING COMPLETED
Best Val Accuracy: 82.45%
Tổng thời gian: ~5 phút
```

Sau đó evaluate trên test set:
```
Overall Accuracy: 0.8134
Weighted F1-Score: 0.8045
```

---

**Ready!** Chỉ cần upload lên Colab + chạy → sẽ có trained model + metrics
