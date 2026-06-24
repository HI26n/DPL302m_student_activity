# Hướng dẫn Thu thập & Xử lý Dữ liệu

Hệ thống nhận diện trạng thái học tập sinh viên — Group 3

---

## Cấu trúc thư mục

```
student_activity/
│
├── setup_project.py        ← Bước 0: chạy 1 lần tạo thư mục
├── extract_frames.py       ← Bước 2: extract + crop từ video
├── review_data.py          ← Bước 3: kiểm tra ảnh bằng mắt
├── check_dataset.py        ← Bước 4: kiểm tra số lượng
├── augment_data.py         ← Bước 5: tăng cường dataset
├── split_dataset.py        ← Bước 6: chia train/val/test
├── requirements.txt        ← Danh sách thư viện cần cài
│
├── videos/                 ← Đặt tất cả file .mov/.mp4 vào đây
│
└── dataset/
    ├── raw/                ← Ảnh sau extract (tự động tạo)
    │   ├── attentive/
    │   ├── phone/
    │   ├── laptop/
    │   ├── sleeping/
    │   └── distracted/
    ├── augmented/          ← Ảnh sau augmentation (tự động tạo)
    │   └── (cấu trúc tương tự)
    ├── split/              ← Ảnh đã chia train/val/test (tự động tạo)
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── review/             ← Ảnh grid để review bằng mắt (tự động tạo)
```

---

## Thứ tự chạy tóm tắt

```
pip install -r requirements.txt      ← 1 lần duy nhất
python setup_project.py              ← 1 lần duy nhất

[Quay video + copy vào videos/]
[Thêm video vào VIDEO_MAP trong extract_frames.py]

python extract_frames.py             ← extract + crop + lọc
python review_data.py                ← kiểm tra ảnh bằng mắt
python check_dataset.py              ← kiểm tra số lượng
python augment_data.py               ← tăng ~150 → ~800 ảnh/class
python split_dataset.py              ← chia 70/15/15
```

---

## Bước 0 — Cài đặt môi trường (làm 1 lần)

### Yêu cầu
- Python 3.10 trở lên
- pip

### Cài thư viện

```bash
pip install -r requirements.txt
```

> Mất khoảng 5–10 phút lần đầu (cài PyTorch + YOLOv8 + OpenCV).

### Tạo thư mục project

```bash
python setup_project.py
```

---

## Bước 1 — Hiểu rõ 5 class

> **Bối cảnh thực tế:** Hầu hết sinh viên trong lớp đều có laptop mở trước mặt.
> Vì vậy, **sự có mặt của laptop KHÔNG phải dấu hiệu phân biệt class**.
> Classifier phân loại dựa trên **hành vi chủ đạo + hướng chú ý** của người đó.

### Định nghĩa 5 class

| Class | Định nghĩa | Dấu hiệu quyết định |
|---|---|---|
| **attentive** | Đang chú ý theo dõi bài giảng | Đầu/mắt hướng về phía **bảng hoặc giảng viên** |
| **laptop** | Đang tương tác với laptop, không theo dõi bài | Mắt nhìn **màn hình laptop**, tay gõ hoặc dùng trackpad |
| **phone** | Đang dùng điện thoại | Tay cầm điện thoại **rõ ràng**, đầu cúi nhìn điện thoại |
| **sleeping** | Đang ngủ | Đầu **gục hẳn** xuống bàn hoặc tựa tay, mắt nhắm |
| **distracted** | Mất tập trung, không chú ý vào bất cứ thứ gì | Đầu/người quay **sang hướng khác**, không nhìn bảng, không nhìn laptop |

### Xử lý các trường hợp overlap

| Tình huống | Class đúng | Lý do |
|---|---|---|
| Laptop mở + đầu nhìn về phía bảng/GV | **attentive** | Hướng chú ý là bài giảng |
| Laptop mở + đang gõ bài | **laptop** | Tương tác chủ động với máy |
| Laptop mở + nhìn màn hình, không gõ | **laptop** | Mắt vẫn trên màn hình |
| Laptop mở + vừa nhìn bảng vừa gõ ghi chép | **attentive** | Hành vi chính là theo dõi bài |
| Laptop mở + cầm điện thoại | **phone** | Phone được ưu tiên hơn laptop |
| Laptop mở + đầu gục xuống bàn | **sleeping** | Sleeping được ưu tiên hơn laptop |
| Laptop mở + quay sang nói chuyện | **distracted** | Không chú ý vào bất cứ thứ gì |

### Quy tắc ưu tiên khi overlap

```
Sleeping > Phone > Laptop > Distracted > Attentive
```

---

## Bước 2 — Quay video

### Phương pháp quay của nhóm

```
- Quay 4 người cùng lúc trong 1 frame
- Tất cả 4 người thực hiện cùng 1 trạng thái
- Mỗi short ~30 giây
- Sau mỗi short: đổi người quay + đổi chỗ ngồi + đổi góc
- Mục tiêu: 4–6 short/class
```

> Cách này hiệu quả hơn quay từng người:
> mỗi short 30s × 4 người × 2fps = ~240 crop thô/short.

### Setup camera

```
Vị trí : Phía trên bảng, nhìn xuống lớp học
Độ cao : 1.5–2m
Góc    : 45–60° so với mặt phẳng ngang
Khoảng cách đến hàng gần nhất: 1.5–3m

Kiểm tra preview trước khi quay:
  ✓ Thấy rõ cả 4 người trong frame
  ✓ Thấy được: đầu + vai + tay + mặt bàn
  ✓ Không ngược sáng
  ✓ Quay landscape (chiều ngang)
```

### Cài đặt quay (iPhone)

```
Độ phân giải : 1080p
FPS          : 30fps (mặc định)
Định dạng    : MOV hoặc MP4 — đều được
Chế độ       : Video thường, KHÔNG dùng slow-motion hay portrait
```

### Nội dung từng class

> **Lưu ý chung:** Laptop mở trên bàn là bình thường ở tất cả các class.
> Không cần dọn laptop đi — để đúng thực tế lớp học.

**ATTENTIVE**
```
Dấu hiệu chính: đầu hướng về phía trước (hướng bảng / GV)

Thay đổi trong short:
  - Nhìn thẳng về phía trước, tay để bàn
  - Vừa nhìn bảng vừa gõ laptop ghi chép
  - Cầm bút ghi chép, thỉnh thoảng ngẩng đầu nhìn bảng
  - Giơ tay lên như muốn phát biểu
  - Gật đầu nhẹ như đang nghe

KHÔNG: cúi mặt nhìn chằm chằm vào màn hình laptop
KHÔNG: cầm điện thoại
```

**LAPTOP**
```
Dấu hiệu chính: mắt nhìn vào màn hình, không nhìn bảng
Laptop MỞ HẲN (góc 90–110°), xoay về phía camera

Thay đổi trong short:
  - Gõ bàn phím liên tục, mắt nhìn màn hình
  - Dừng gõ, tay đặt lên trackpad, vẫn nhìn màn hình
  - Một tay chống cằm, nhìn màn hình
  - Cuộn chuột, đọc nội dung trên màn hình

KHÔNG: ngẩng đầu nhìn về phía bảng/GV (→ attentive)
KHÔNG: cầm điện thoại đồng thời
```

**PHONE**
```
Dấu hiệu chính: tay cầm điện thoại rõ ràng
Laptop có thể để trên bàn — không ảnh hưởng
Điện thoại PHẢI THẤY RÕ từ góc camera — kiểm tra preview

Thay đổi trong short:
  - Cầm điện thoại 2 tay, đặt trước màn hình laptop, cúi nhìn
  - Cầm điện thoại 1 tay bên cạnh laptop
  - Điện thoại để dưới mép bàn, cúi xuống nhìn
  - Thỉnh thoảng ngẩng đầu lên rồi cúi xuống lại
```

**SLEEPING**
```
Dấu hiệu chính: đầu gục hẳn xuống, mắt nhắm
Laptop có thể để mở hoặc đóng lại trên bàn

Thay đổi trong short:
  - Đầu gục xuống bàn, tay làm gối
  - Tựa đầu vào tay, mắt nhắm
  - Đầu nghiêng sang bên
  - Ngả người ra sau, đầu ngửa
```

**DISTRACTED**
```
Dấu hiệu chính: đầu/người quay sang hướng khác
Laptop có thể để trên bàn — không ảnh hưởng
KHÔNG cầm điện thoại, KHÔNG nhìn màn hình laptop

Thay đổi trong short:
  - Quay sang bên trái như nói chuyện với bạn
  - Quay sang bên phải
  - Nhìn ra cửa sổ hoặc hướng khác
  - Ngồi vặn vẹo, chống cằm nhìn lơ đễnh
  - Quay người ra sau
```

### Đặt tên file video

```
[class]_[số].mov   hoặc   [class]_[số].mp4

Ví dụ:
  attentive_01.mov
  attentive_02.mov
  phone_01.mov
  laptop_01.mov
  sleeping_01.mov
  distracted_01.mov
```

### Chuyển video về máy tính

```
Cách 1 — AirDrop (nhanh nhất với iPhone + Mac):
  AirDrop video sang Mac → copy vào videos/

Cách 2 — Cáp USB:
  Cắm iPhone → mở Finder (Mac) hoặc iTunes (Windows)
  → copy .mov vào videos/

Cách 3 — Telegram Saved Messages:
  Gửi video vào Saved Messages → Download trên máy tính
  → đặt vào videos/

Lưu ý: iPhone xuất file .mov — OpenCV đọc được bình thường.
```

---

## Bước 3 — Chạy extract

### Thêm video vào VIDEO_MAP

Mở file `extract_frames.py` bằng VS Code.
Tìm phần `VIDEO_MAP`, bỏ dấu `#` ở các dòng tương ứng:

```python
VIDEO_MAP = [
    ("videos/attentive_01.mov",  "dataset/raw/attentive"),
    ("videos/attentive_02.mov",  "dataset/raw/attentive"),
    ("videos/phone_01.mov",      "dataset/raw/phone"),
    # thêm tiếp...
]
```

### Chạy

```bash
python extract_frames.py
```

**Lần đầu chạy:** YOLOv8n tự download ~6MB (cần internet).

**Kết quả terminal mẫu:**

```
📹 attentive_01.mov [iPhone MOV]
   Class   : attentive
   Dài     : 32s | 30fps
   Dự kiến : ~256 crop (4 người × 64 frame)
   ✅ Lưu được : 198 ảnh
   ❌ Bỏ qua   : 24 (nhỏ=4 | mép=6 | mờ=9 | che=5)
   📊 Giữ lại  : 89.2%
   ✅ Tổng attentive: 198 ảnh — gần đủ, cần thêm 52 ảnh nữa
```

**Nếu tỉ lệ giữ lại < 60%:** Camera đặt quá xa hoặc góc không tốt.

---

## Bước 4 — Review ảnh bằng mắt

```bash
python review_data.py
```

Mở thư mục `dataset/review/`, xem từng file `review_<class>.jpg`.

**Xóa ảnh nếu:**
```
✗ Chỉ thấy mặt, không thấy tay hay đồ vật
✗ Ảnh bị mờ hoặc bị che quá nhiều
✗ Người đang ở trạng thái chuyển tiếp giữa 2 class
✗ Bị cắt mất phần quan trọng

Riêng class laptop:
✗ Người đang ngẩng đầu nhìn bảng (→ nên là attentive)

Riêng class attentive:
✗ Mắt cúi nhìn chằm chằm màn hình laptop (→ nên là laptop)
```

Xóa thủ công trong `dataset/raw/<class>/` rồi tiếp tục.

---

## Bước 5 — Kiểm tra số lượng

```bash
python check_dataset.py
```

Mục tiêu: mỗi class **≥ 150 ảnh gốc** trước khi augment.

---

## Bước 6 — Augmentation

```bash
python augment_data.py
```

Tăng từ ~150 ảnh gốc/class lên **~800 ảnh/class**.

Các transform được thiết kế cho lớp học thực tế:

| Transform | Mục đích |
|---|---|
| Flip ngang | Lớp học đối xứng hai bên |
| Xoay ±8° | Camera đặt không hoàn toàn thẳng |
| Perspective nhỏ | Đổi góc, đổi chỗ giữa các short |
| Brightness/Contrast | Ánh sáng lớp học thay đổi |
| Color temperature | Đèn huỳnh quang / đèn vàng / ánh sáng tự nhiên |
| Blur nhẹ | Motion blur, người ngồi xa hơi mờ |
| Noise nhẹ | Compression video |

Kết quả lưu vào `dataset/augmented/`.

---

## Bước 7 — Chia train/val/test

```bash
python split_dataset.py
```

Tỉ lệ chia: **70% train / 15% val / 15% test**

> **Lưu ý quan trọng — tránh data leakage:**
> - Ảnh **augment** → chỉ vào **train**
> - Ảnh **gốc** → chỉ vào **val** và **test**
>
> Nếu ảnh gốc và ảnh augment từ cùng nguồn lẫn vào cả train lẫn
> val/test, model sẽ "nhớ" ảnh gốc và cho kết quả val/test cao giả tạo.

Kết quả mẫu:

```
══════════════════════════════════════════════════════════
              KẾT QUẢ SPLIT DATASET
══════════════════════════════════════════════════════════
  Class           Train     Val    Test   Total
  ──────────────────────────────────────────────────────
  attentive         560      80      80     720
  phone             560      80      80     720
  laptop            560      80      80     720
  sleeping          560      80      80     720
  distracted        560      80      80     720
  ──────────────────────────────────────────────────────
  TOTAL            2800     400     400    3600

  → Bước tiếp theo: python train.py
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| `ModuleNotFoundError: ultralytics` | Chưa cài thư viện | `pip install -r requirements.txt` |
| `Không tìm thấy: videos/xxx.mov` | Tên file sai hoặc chưa copy vào | Kiểm tra tên file trong `videos/` |
| Tỉ lệ giữ lại < 50% | Camera quá xa hoặc góc sai | Quay lại, đặt camera gần hơn |
| Ảnh chỉ có mặt, không có bàn | PAD_BOT chưa đủ | Tăng `PAD_BOT` lên 1.0 trong `extract_frames.py` |
| Video MOV không mở được | Codec lạ | Dùng VLC convert sang MP4 trước |
| `CUDA out of memory` | GPU không đủ RAM | Thêm `device='cpu'` vào `YOLO()` |
