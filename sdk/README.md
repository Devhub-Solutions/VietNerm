
# DISCLAIMER & ETHICAL USE #
```shell
Dự án này được xây dựng phục vụ nghiên cứu AI và xử lý tài liệu (Document AI).

Tất cả dữ liệu được sinh ra trong dự án này hoàn toàn là dữ liệu giả lập (synthetic data),
chỉ phục vụ mục đích nghiên cứu và phát triển hệ thống AI.

Các nguyên tắc quan trọng:

1. Dữ liệu giả lập

Tất cả dữ liệu trong dự án:

- Không sử dụng dữ liệu cá nhân thật
- Không sử dụng giấy tờ thật
- Không thu thập thông tin người dùng

Dữ liệu được sinh hoàn toàn từ hệ thống generator.


2. An toàn số định danh (ID)

Hệ thống sinh dữ liệu được thiết kế để không trùng với dữ liệu thật.

Các nguyên tắc:

- Các số định danh được sinh ngẫu nhiên
- Kiểm tra xác suất để tránh trùng với quy luật số thật
- Ưu tiên sử dụng các đầu số chưa được cấp phát

Ví dụ ID giả lập:

Mock ID: 900988214412


3. Hình ảnh giấy tờ

Nếu sử dụng hệ thống render mockup giấy tờ:

- Luôn có watermark "SAMPLE" hoặc "MOCKUP"
- Không chia sẻ hình ảnh giống giấy tờ thật lên các nền tảng công cộng
  nếu không có watermark

Ví dụ watermark:

SAMPLE
MOCKUP
SYNTHETIC DATA


4. Mục đích sử dụng

Dự án được xây dựng cho:

- Research AI
- Document AI
- OCR / NER pipeline
- Synthetic dataset generation

Không được sử dụng cho:

- Giả mạo giấy tờ
- Tạo giấy tờ giả
- Lừa đảo hoặc gian lận
```

# VietNerm - Document AI Factory cho Giấy tờ Việt Nam

Hệ thống AI pipeline hoàn chỉnh để trích xuất thực thể (NER) từ các loại giấy tờ tiếng Việt, sử dụng kiến trúc PhoBERT.

**Pipeline**: Template → Synthetic Data → NER Dataset → Training → Model → HuggingFace → SDK

## Kiến trúc tổng thể

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐
│  Templates   │───▶│  Synthetic   │───▶│   Dataset    │───▶│ Training  │
│  (Jinja2)    │    │  Generator   │    │  (BIO NER)   │    │ (PhoBERT) │
└─────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘
                                                                  │
                   ┌──────────────┐    ┌──────────────┐          │
                   │     SDK      │◀───│  HuggingFace │◀─────────┘
                   │  (VietNerm)  │    │    Hub       │    ┌───────────┐
                   └──────┬───────┘    └──────────────┘    │  Models/  │
                          │                                └───────────┘
                          ▼
                   ┌──────────────┐
                   │  Inference   │──▶ {"name": "...", "dob": "..."}
                   │  Pipeline    │
                   └──────────────┘
```

## Cấu trúc dự án

```
vietnerm/
├── registry/                      # Registry document types
│   └── documents.yaml
├── templates/                     # Jinja2 templates cho từng loại giấy tờ
│   ├── cccd/
│   │   ├── schema.yaml
│   │   └── template_*.txt
│   └── giay_ra_vien/
│       ├── schema.yaml
│       └── template_*.txt
├── synthetic/                     # Sinh dữ liệu tổng hợp
│   ├── generators/                # Data generators (person, hospital, vehicle)
│   ├── template_engine.py         # Render Jinja2 templates
│   ├── noise_engine.py            # Giả lập lỗi OCR
│   └── generate_dataset.py        # CLI entry point
├── datasets/                      # Dữ liệu train/test
│   ├── raw/{doc}/                 # Raw synthetic records
│   └── ner/{doc}/                 # BIO-tagged NER data
├── training/                      # Training pipeline
│   ├── config/                    # Training configs per doc type
│   ├── trainer.py                 # PhoBERT NER trainer
│   └── train.py                   # Universal training CLI
├── models/                        # Trained model output
├── inference/                     # Inference pipeline
│   ├── pipeline.py                # NERPipeline: load model, predict
│   ├── schema_mapper.py           # Map BIO → structured dict
│   └── postprocess.py             # Sub-token merging, validation
├── huggingface/                   # Publish to HuggingFace Hub
│   ├── push_model.py              # Upload model
│   ├── push_dataset.py            # Upload dataset
│   └── model_card.md              # Model card template
├── sdk/vietnerm/                  # Python SDK package
│   ├── __init__.py
│   ├── ner.py                     # VietNerm, CCCDNer, GiayRaVienNer
│   └── utils.py                   # Tiện ích xử lý text
├── scripts/                       # Shell scripts
│   ├── generate_data.sh
│   ├── train.sh
│   └── publish.sh
├── docs/                          # Tài liệu
│   ├── architecture.md
│   └── adding_new_document.md
├── raw_code/                      # Code gốc (tham khảo)
├── requirements.txt
└── .github/workflows/ci.yml       # CI/CD
```

## Cài đặt

### Yêu cầu
- Python 3.9+
- PyTorch 1.10+
- CUDA (khuyến nghị, không bắt buộc)

### Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Cài đặt SDK (development mode)

```bash
pip install -e ./sdk/
```

## Bắt đầu nhanh

### 1. Sinh dữ liệu tổng hợp

```bash
# Sinh 10,000 mẫu cho CCCD
./scripts/generate_data.sh cccd 10000

# Sinh 50,000 mẫu cho Giấy ra viện
./scripts/generate_data.sh giay_ra_vien 50000
```

### 2. Huấn luyện mô hình

```bash
# Train model cho CCCD
./scripts/train.sh cccd

# Train model cho Giấy ra viện với custom params
./scripts/train.sh giay_ra_vien --epochs 10 --batch_size 32
```

### 3. Trích xuất thực thể

```python
from vietnerm import VietNerm

# Tạo extractor cho CCCD
ner = VietNerm(doc_type="cccd")
result = ner.extract("""
Họ và tên: NGUYỄN VĂN A
Ngày sinh: 01/01/1990
Giới tính: Nam
Quốc tịch: Việt Nam
""")
print(result)
# {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}
```

### 4. Publish lên HuggingFace

```bash
./scripts/publish.sh cccd username
```

## Pipeline chi tiết

### Bước 1: Định nghĩa Document Type

Tạo template và schema trong `templates/{doc_type}/`:

```yaml
# templates/cccd/schema.yaml
doc_type: cccd
entities:
  - name: name
    type: person_name
  - name: date_of_birth
    type: date
  - name: id_number
    type: id
```

### Bước 2: Sinh dữ liệu

```bash
python synthetic/generate_dataset.py --doc cccd --size 10000
```

Pipeline tự động:
1. Load template Jinja2
2. Sinh dữ liệu ngẫu nhiên (tên, ngày sinh, địa chỉ...)
3. Render template → text + character spans
4. Inject nhiễu OCR
5. Convert sang BIO NER format
6. Split train/test (80/10/10)

### Bước 3: Huấn luyện

```bash
python training/train.py --doc cccd --epochs 7
```

Sử dụng PhoBERT (`vinai/phobert-base`) fine-tuned cho NER token classification.

### Bước 4: Inference

```python
from inference.pipeline import NERPipeline
from inference.schema_mapper import SchemaMapper

pipeline = NERPipeline(model_path="models/phobert/cccd/")
raw_entities = pipeline.predict("Họ và tên: NGUYỄN VĂN A...")

mapper = SchemaMapper(doc_type="cccd")
result = mapper.map_entities(raw_entities)
```

## Thêm loại giấy tờ mới

Xem hướng dẫn chi tiết: [docs/adding_new_document.md](docs/adding_new_document.md)

Tóm tắt:
1. Thêm entry vào `registry/documents.yaml`
2. Tạo `templates/{doc_type}/schema.yaml`
3. Tạo template files `templates/{doc_type}/template_*.txt`
4. Thêm generator vào `synthetic/generators/`
5. Thêm training config `training/config/{doc_type}.yaml`
6. Sinh dữ liệu → Train → Publish

## SDK API Reference

### VietNerm

```python
from vietnerm import VietNerm

# Tạo extractor
ner = VietNerm(doc_type="cccd")
ner = VietNerm(doc_type="cccd", model_path="path/to/model")
ner = VietNerm(doc_type="cccd", device="cuda")

# Trích xuất
result = ner.extract(text)                    # Dict[str, str]
result = ner.extract_with_confidence(text)    # Dict[str, {"value": str, "confidence": float}]
raw = ner.extract_raw(text)                   # List[Dict] - raw entities
```

### Cấu hình download model (SSL / cache / predownload)

```python
from vietnerm import VietNerm, DownloadConfig

cfg = DownloadConfig(
    cache_dir="./.hf-cache",      # custom cache
    disable_ssl_verify=True,       # tắt verify SSL (mạng nội bộ/self-signed)
    force_download=False,
)

ner = VietNerm(doc_type="cccd", download_config=cfg)

# Predownload model để chạy offline / giảm cold-start
local_snapshot = VietNerm.predownload("cccd", download_config=cfg)

# Xóa cache model cụ thể
VietNerm.clear_model_cache(repo_id="ngocthanhdoan/phobert-cccd-ner")

# Hoặc xóa toàn bộ cache HF
VietNerm.clear_model_cache(cache_dir="./.hf-cache")
```

### Inference Pipeline (low-level)

```python
from inference.pipeline import NERPipeline
from inference.schema_mapper import SchemaMapper

# Load pipeline
pipeline = NERPipeline(model_path="models/phobert/cccd/")
raw_entities = pipeline.predict(text)

# Map to structured output
mapper = SchemaMapper(doc_type="cccd")
result = mapper.map_entities(raw_entities)
```

## Các loại giấy tờ được hỗ trợ

| Loại giấy tờ | Doc Type | Entities |
|---|---|---|
| Căn cước công dân | `cccd` | Số CCCD, Họ tên, Ngày sinh, Giới tính, Quốc tịch, Quê quán, Nơi thường trú, Hạn sử dụng |
| Giấy ra viện | `giay_ra_vien` | Bệnh viện, Khoa, Mã y tế, Tên BN, Ngày sinh, Giới tính, Dân tộc, Nghề nghiệp, Địa chỉ, BHXH, Ngày vào/ra viện, Chẩn đoán, Phương pháp điều trị, Ghi chú |
| Đăng ký xe | `vehicle_registration` | Tên chủ sở hữu, Biển số, Loại xe, Hãng, Số máy, Số khung |

## Đóng góp

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Code theo conventions:
   - Type hints cho tất cả functions
   - Docstrings (Google style)
   - UTF-8 cho text tiếng Việt
   - `pathlib` cho đường dẫn file
4. Chạy lint: `flake8 . --max-line-length=120`
5. Tạo Pull Request

## License
```
Copyright (c) 2026 Devhub Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction.
```
