"""
extract_frames.py
=================
Đọc video từ thư mục videos/, dùng YOLOv8 phát hiện người,
crop thông minh, lọc ảnh kém chất lượng, lưu vào dataset/raw/<class>/.

Thiết kế cho:
  - Góc camera: phía trên bảng nhìn xuống lớp (~45–60°)
  - Quay 4 người cùng lúc trong 1 frame
  - Video từ iPhone (MOV hoặc MP4)
  - Mỗi short ~30s, nhiều short/class

Cách chạy:
    python extract_frames.py

Trước khi chạy:
    1. Đặt video vào thư mục videos/
    2. Thêm video vào VIDEO_MAP bên dưới (bỏ dấu #)

Quy tắc đặt tên video:
    [class]_[số].mov  hoặc  [class]_[số].mp4
    Ví dụ: attentive_01.mov | phone_02.mov | laptop_01.mp4
"""

import cv2
import os
import numpy as np
from ultralytics import YOLO

# ═══════════════════════════════════════════════════════════
#  CẤU HÌNH — đã tinh chỉnh cho góc từ trên bảng nhìn xuống
# ═══════════════════════════════════════════════════════════

FPS_EXTRACT   = 2      # Số frame lấy mỗi giây
CONF_THRESH   = 0.25   # Giảm nhẹ so với mặc định vì góc trên xuống
                       # đôi khi YOLO ít tự tin hơn với góc lạ
MIN_BOX_RATIO = 0.03   # Giảm xuống 3% vì 4 người/frame → mỗi người nhỏ hơn 
BLUR_THRESH   = 60     # Giảm nhẹ, iPhone thường nét hơn Android
FRAME_MARGIN  = 15     # Pixel cách mép frame

# ── Padding crop — góc từ trên bảng nhìn xuống ─────────────
# Góc này thấy: đỉnh đầu → mặt → vai → tay → mặt bàn
# Cần lấy đủ vùng bàn để thấy laptop/điện thoại
#
#  YOLO box thường bao: [đỉnh đầu → thắt lưng]
#  Ta mở rộng:
#    - Ngang: 20% mỗi bên (vai rộng hơn từ góc trên)
#    - Trên:  5%  (đỉnh đầu thường vào frame đủ)
#    - Dưới:  80% (lấy vùng bàn + tay + laptop/điện thoại)
#
PAD_X   = 0.10   # Mở rộng ngang mỗi bên 20%
PAD_TOP = 0.05   # Mở rộng lên trên 5%
PAD_BOT = 0.40   # Mở rộng xuống 80% — đủ lấy mặt bàn góc này

TARGET_SIZE = 224  # Kích thước đầu vào MobileNetV2

# ═══════════════════════════════════════════════════════════
#  VIDEO MAP
#  - Hỗ trợ cả .mov (iPhone) lẫn .mp4
#  - Bỏ dấu # ở dòng tương ứng sau khi có video
# ═══════════════════════════════════════════════════════════

VIDEO_MAP = [
    # ── ATTENTIVE ──────────────────────────────────────────
    ("videos/attentive_1.mp4",   "dataset/raw/attentive"),
    ("videos/attentive_2.mov",   "dataset/raw/attentive"),
    ("videos/attentive_3.mov",   "dataset/raw/attentive"),
    # ("videos/attentive_04.mov",   "dataset/raw/attentive"),
    # ("videos/attentive_05.mov",   "dataset/raw/attentive"),
    # ("videos/attentive_06.mov",   "dataset/raw/attentive"),

    # ── PHONE ──────────────────────────────────────────────
    ("videos/phone_1.mp4",       "dataset/raw/phone"),
    ("videos/phone_2.mov",       "dataset/raw/phone"),
    ("videos/phone_3.mov",       "dataset/raw/phone"),
    # ("videos/phone_04.mov",       "dataset/raw/phone"),
    # ("videos/phone_05.mov",       "dataset/raw/phone"),
    # ("videos/phone_06.mov",       "dataset/raw/phone"),

    # ── LAPTOP ─────────────────────────────────────────────
    ("videos/laptop_1.mp4",      "dataset/raw/laptop"),
    ("videos/laptop_2.mov",      "dataset/raw/laptop"),
    ("videos/laptop_3.mov",      "dataset/raw/laptop"),
    # ("videos/laptop_04.mov",      "dataset/raw/laptop"),
    # ("videos/laptop_05.mov",      "dataset/raw/laptop"),
    # ("videos/laptop_06.mov",      "dataset/raw/laptop"),

    # ── SLEEPING ───────────────────────────────────────────
    ("videos/sleep_1.mp4",    "dataset/raw/sleeping"),
    ("videos/sleep_2.mov",    "dataset/raw/sleeping"),
    ("videos/sleep_3.mov",    "dataset/raw/sleeping"),
    # ("videos/sleeping_04.mov",    "dataset/raw/sleeping"),
    # ("videos/sleeping_05.mov",    "dataset/raw/sleeping"),
    # ("videos/sleeping_06.mov",    "dataset/raw/sleeping"),

    # ── DISTRACTED ─────────────────────────────────────────
    ("videos/distract_1.mp4",  "dataset/raw/distracted"),
    ("videos/distract_2.mov",  "dataset/raw/distracted"),
    ("videos/distract_3.mov",  "dataset/raw/distracted")
    # ("videos/distracted_04.mov",  "dataset/raw/distracted"),
    # ("videos/distracted_05.mov",  "dataset/raw/distracted"),
    # ("videos/distracted_06.mov",  "dataset/raw/distracted"),
]


# ═══════════════════════════════════════════════════════════
#  CHUYỂN ĐỔI MOV → MP4 (iPhone)
#  iPhone quay ra .mov — OpenCV đọc được nhưng đôi khi
#  bị lỗi rotation metadata. Hàm này xử lý tự động.
# ═══════════════════════════════════════════════════════════

def fix_iphone_rotation(frame, cap):
    """
    iPhone ghi metadata rotation vào video nhưng không xoay frame thật.
    Hàm này đọc rotation từ metadata và xoay frame cho đúng.
    OpenCV không đọc được EXIF trực tiếp nên ta dùng heuristic:
    nếu frame cao hơn rộng (portrait) thì xoay 90°.
    """
    h, w = frame.shape[:2]
    if h > w:
        # Video quay dọc (portrait) → xoay thành ngang
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


# ═══════════════════════════════════════════════════════════
#  HÀM XỬ LÝ ẢNH
# ═══════════════════════════════════════════════════════════

def smart_crop(frame, box):
    """
    Crop mở rộng cho góc từ trên bảng nhìn xuống.

    YOLO từ góc này thường detect box bao [đầu → thắt lưng].
    Ta mở rộng ngang (vai rộng hơn) và xuống dưới (lấy mặt bàn
    để thấy laptop, điện thoại, sách vở).

    PAD_BOT = 0.80 thay vì 1.00 vì góc này mặt bàn đã gần hơn
    trong box so với góc ngang.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bh = y2 - y1
    bw = x2 - x1

    nx1 = max(0, x1 - int(bw * PAD_X))
    ny1 = max(0, y1 - int(bh * PAD_TOP))
    nx2 = min(w, x2 + int(bw * PAD_X))
    ny2 = min(h, y2 + int(bh * PAD_BOT))

    crop = frame[ny1:ny2, nx1:nx2]
    return crop if crop.size > 0 else None


def letterbox_resize(img, size=224):
    """
    Resize về 224×224 giữ nguyên tỉ lệ, fill xám phần thừa.
    Tránh méo hình — quan trọng cho classifier.
    """
    h, w    = img.shape[:2]
    scale   = size / max(h, w)
    nh, nw  = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas  = np.full((size, size, 3), 128, dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# ─── Bộ lọc chất lượng ─────────────────────────────────────

def filter_box_too_small(box, frame_shape):
    """Loại người quá nhỏ. Ngưỡng 3% vì 4 người/frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    ratio = ((x2 - x1) * (y2 - y1)) / (h * w)
    return ratio >= MIN_BOX_RATIO


def filter_box_clipped(box, frame_shape):
    """Loại người bị cắt bởi mép frame."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    return (
        x1 > FRAME_MARGIN and
        y1 > FRAME_MARGIN and
        x2 < w - FRAME_MARGIN and
        y2 < h - FRAME_MARGIN
    )


def filter_blurry(crop):
    """Loại ảnh mờ. Ngưỡng 60 vì iPhone thường nét."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESH


def filter_occluded(crop):
    """Loại ảnh bị che khuất quá nhiều."""
    gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blank = (gray < 20) | (gray > 235)
    return (blank.sum() / gray.size) < 0.5


def count_images(folder):
    """Đếm ảnh hiện có để đặt tên nối tiếp."""
    if not os.path.exists(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png"))
    ])


# ═══════════════════════════════════════════════════════════
#  XỬ LÝ 1 VIDEO
# ═══════════════════════════════════════════════════════════

def process_video(video_path, output_dir, model):
    if not os.path.exists(video_path):
        print(f"\n  ⚠️  Không tìm thấy: {video_path}")
        print(     "     Kiểm tra tên file và đường dẫn trong VIDEO_MAP.")
        return

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"\n  ❌ Không mở được video: {video_path}")
        print(     "     Nếu là file .mov từ iPhone, thử đổi sang .mp4")
        print(     "     bằng cách AirDrop sang Mac rồi export, hoặc")
        print(     "     dùng VLC: Media → Convert/Save → chọn MP4.")
        return

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / video_fps if video_fps > 0 else 0
    interval     = max(1, int(video_fps / FPS_EXTRACT))

    start_idx = count_images(output_dir)
    idx       = start_idx
    skipped   = {"small": 0, "clipped": 0, "blurry": 0, "occluded": 0}
    frame_no  = 0
    is_iphone = video_path.lower().endswith(".mov")

    cls_name = os.path.basename(output_dir)
    print(f"\n  {'─'*52}")
    print(f"  📹 {os.path.basename(video_path)}"
          + (" [iPhone MOV]" if is_iphone else ""))
    print(f"     Class   : {cls_name}")
    print(f"     Dài     : {duration:.0f}s | {video_fps:.0f}fps")
    print(f"     Dự kiến : ~{int(duration * FPS_EXTRACT * 4)} crop "
          f"(4 người × {int(duration * FPS_EXTRACT)} frame)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý rotation cho iPhone
        if is_iphone:
            frame = fix_iphone_rotation(frame, cap)

        if frame_no % interval == 0:
            results = model(
                frame,
                classes=[0],
                conf=CONF_THRESH,
                verbose=False
            )
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:
                if not filter_box_too_small(box, frame.shape):
                    skipped["small"] += 1
                    continue
                if not filter_box_clipped(box, frame.shape):
                    skipped["clipped"] += 1
                    continue

                crop = smart_crop(frame, box)
                if crop is None:
                    continue

                if not filter_blurry(crop):
                    skipped["blurry"] += 1
                    continue
                if not filter_occluded(crop):
                    skipped["occluded"] += 1
                    continue

                final = letterbox_resize(crop, TARGET_SIZE)
                fname = f"{cls_name}_{idx:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), final)
                idx += 1

        frame_no += 1

    cap.release()

    new_saved   = idx - start_idx
    total_tried = new_saved + sum(skipped.values())

    print(f"     ✅ Lưu được : {new_saved} ảnh")
    if total_tried > 0:
        skip_total = sum(skipped.values())
        print(
            f"     ❌ Bỏ qua   : {skip_total} "
            f"(nhỏ={skipped['small']} | "
            f"mép={skipped['clipped']} | "
            f"mờ={skipped['blurry']} | "
            f"che={skipped['occluded']})"
        )
        print(f"     📊 Giữ lại  : {new_saved/total_tried*100:.1f}%")

    # Cảnh báo nếu ảnh ít
    total_now = count_images(output_dir)
    print(f"     Tổng {cls_name}: {total_now}")


# ═══════════════════════════════════════════════════════════
#  THỐNG KÊ DATASET
# ═══════════════════════════════════════════════════════════

def print_dataset_summary():
    CLASSES = ["attentive", "phone", "laptop", "sleeping", "distracted"]
    TARGET  = 150

    print(f"\n  {'═'*52}")
    print(f"  {'THỐNG KÊ DATASET':^52}")
    print(f"  {'═'*52}")
    print(f"  {'Class':<15} {'Ảnh':>6}  {'Trạng thái'}")
    print(f"  {'─'*48}")

    total  = 0
    all_ok = True
    for cls in CLASSES:
        count = count_images(f"dataset/raw/{cls}")
        total += count

        if count >= TARGET:
            status = "✅ Đủ"
        elif count >= TARGET // 2:
            status = f"⚠️  Cần thêm {TARGET - count} ảnh"
            all_ok = False
        else:
            status = f"❌ Thiếu — cần thêm {TARGET - count} ảnh"
            all_ok = False

        print(f"  {cls:<15} {count:>6}  {status}")

    print(f"  {'─'*48}")
    print(f"  {'TOTAL':<15} {total:>6}")
    print(f"  {'═'*52}")

    if all_ok:
        print("\n  ✅ Dataset đủ! Chạy tiếp: python review_data.py")
    else:
        print("\n  ⚠️  Cần thêm video cho class thiếu.")
        print("     Thêm vào VIDEO_MAP rồi chạy lại file này.\n")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    active = [x for x in VIDEO_MAP]
    if not active:
        print("\n⚠️  VIDEO_MAP đang trống!")
        print("   Mở file này, tìm VIDEO_MAP,")
        print("   bỏ dấu # ở các dòng tương ứng với video đã có.\n")
        print_dataset_summary()
        exit(0)

    print("\n🔄 Đang tải YOLOv8n...")
    print("   (Lần đầu tự download ~6MB, cần internet)\n")
    model = YOLO("yolov8n.pt")
    print("✅ Sẵn sàng!\n")

    for video_path, output_dir in active:
        process_video(video_path, output_dir, model)

    print_dataset_summary()
