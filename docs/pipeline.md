# Pipeline Tổng thể

## Sơ đồ

```
┌─────────────────────────────────────────────────────────────┐
│  DATA COLLECTION (scripts/data/)                            │
├─────────────────────────────────────────────────────────────┤
│  videos/ → extract → review → check → augment             │
│         → label_persons → split_by_person                   │
│                          ↓                                  │
│              dataset/split_by_person/                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  MODELING (scripts/model/)                                  │
├─────────────────────────────────────────────────────────────┤
│  eda → check_integrity → train → evaluate                    │
│                          ↓                                  │
│              models/best_model.pth                          │
│              logs/training/ + logs/evaluation/              │
└─────────────────────────────────────────────────────────────┘
```

## Bảng tóm tắt

| Bước | Script | Output |
|---|---|---|
| Setup | `setup_project.py` | Cấu trúc thư mục |
| Extract | `scripts/data/extract_frames.py` | `dataset/raw/` |
| Review | `scripts/data/review_data.py` | `dataset/review/` |
| Check | `scripts/data/check_dataset.py` | Terminal report |
| Augment | `scripts/data/augment_data.py` | `dataset/augmented/` |
| Label | `scripts/data/label_persons.py` | Nhãn p1-p5 |
| Split | `scripts/data/split_by_person.py` | `dataset/split_by_person/` |
| EDA | `scripts/model/eda.py` | `logs/eda/` |
| Integrity | `scripts/model/check_integrity.py` | Terminal report |
| Train | `scripts/model/train.py` | `models/best_model.pth` |
| Evaluate | `scripts/model/evaluate.py` | `logs/evaluation/` |

## Split theo người

```
Train : aug p1, p2, p3, p4
Val   : 50% raw (p1 + p5)
Test  : 50% raw (p1 + p5)

Chống leakage: p1 chia theo raw_idx — aug train ≠ raw val/test
```

## Logs

```
logs/
├── eda/           ← eda.py
├── training/      ← train.py
└── evaluation/    ← evaluate.py
```

## Báo cáo nhận xét

Xem `docs/reports/`:
- `nhan_xet_eda.txt`
- `nhan_xet_train.txt`
- `nhan_xet_evaluate.txt`
