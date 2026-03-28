# VietNerm — Document AI Factory cho Giấy tờ Việt Nam

> **Hệ thống AI pipeline hoàn chỉnh** để trích xuất thực thể (NER) từ các loại giấy tờ tiếng Việt, sử dụng kiến trúc PhoBERT.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PhoBERT](https://img.shields.io/badge/model-PhoBERT-orange.svg)](https://huggingface.co/vinai/phobert-base)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/ngocthanhdoan)

---

## ⚠️ Disclaimer & Ethical Use

> Dự án này được xây dựng **phục vụ nghiên cứu AI và xử lý tài liệu (Document AI)**. Tất cả dữ liệu là **synthetic data** hoàn toàn giả lập, không liên quan đến thông tin cá nhân thật.

| Nguyên tắc | Mô tả |
|---|---|
| 🔒 Dữ liệu giả lập | Không sử dụng dữ liệu cá nhân hoặc giấy tờ thật |
| 🆔 An toàn số định danh | Số ID sinh ngẫu nhiên, tránh trùng với dữ liệu thật |
| 🖼️ Watermark bắt buộc | Hình ảnh mockup luôn có `SAMPLE` / `SYNTHETIC DATA` |
| ✅ Mục đích hợp lệ | Research AI, OCR pipeline, NER dataset generation |
| ❌ Cấm sử dụng | Giả mạo giấy tờ, lừa đảo, tạo giấy tờ giả |

---

## 📋 Mục lục

1. [Tổng quan](#-tổng-quan)
2. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
3. [Cài đặt](#-cài-đặt)
4. [Cấu trúc dự án](#-cấu-trúc-dự-án)
5. [Pipeline chi tiết](#-pipeline-chi-tiết)
6. [Templates & Registry](#-templates--registry)
7. [Training trên Kaggle GPU](#-training-trên-kaggle-gpu)
8. [Kết quả Training](#-kết-quả-training)
9. [Publish lên HuggingFace](#-publish-lên-huggingface)
10. [SDK API Reference](#-sdk-api-reference)
11. [Các loại giấy tờ hỗ trợ](#-các-loại-giấy-tờ-hỗ-trợ)
12. [Đóng góp](#-đóng-góp)

---

## 🧠 Tổng quan

**VietNerm** là một Document AI Factory giúp tự động trích xuất thực thể có tên (Named Entity Recognition) từ các loại giấy tờ hành chính Việt Nam.

**Pipeline tổng thể:**

```
Template → Synthetic Data → NER Dataset → Training → Model → HuggingFace → SDK
```

**Các loại giấy tờ được hỗ trợ:**
- 🪪 Căn cước công dân (CCCD)
- 🏥 Giấy ra viện
- 🚗 Đăng ký xe
- 🪪 Giấy phép lái xe (GPLX)
- 📄 Giấy khai sinh

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐
│  Templates   │───▶│  Synthetic   │───▶│   Dataset    │───▶│ Training  │
│  (Jinja2)    │    │  Generator   │    │  (BIO NER)   │    │ (PhoBERT) │
└─────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘
                                                                  │
                   ┌──────────────┐    ┌──────────────┐          │
                   │     SDK      │◀───│  HuggingFace │◀─────────┘
                   │  (VietNerm)  │    │    Hub       │
                   └──────┬───────┘    └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Inference   │──▶ {"name": "...", "dob": "..."}
                   │  Pipeline    │
                   └──────────────┘
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống

| Thành phần | Phiên bản |
|---|---|
| Python | 3.9+ |
| PyTorch | 1.10+ |
| CUDA | Khuyến nghị (không bắt buộc) |

### Bước 1: Clone repository

```bash
git clone https://github.com/Devhub-Solutions/VietNerm.git
cd VietNerm
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 3: Cài đặt SDK (development mode)

```bash
pip install -e ./sdk/
```

### Bước 4: Kiểm tra cài đặt

```python
from vietnerm import VietNerm
print("VietNerm installed successfully!")
```

---

## 📁 Cấu trúc dự án

```
vietnerm/
├── registry/                      # Registry document types
│   └── documents.yaml
├── templates/                     # Jinja2 templates cho từng loại giấy tờ
│   ├── cccd/
│   │   ├── schema.yaml
│   │   └── template_*.txt
│   ├── giay_khai_sinh/
│   ├── giay_ra_vien/
│   ├── gplx/
│   └── vehicle_registration/
├── synthetic/                     # Sinh dữ liệu tổng hợp
│   ├── generators/
│   ├── template_engine.py
│   ├── noise_engine.py
│   └── generate_dataset.py
├── datasets/                      # Dữ liệu train/test
│   ├── raw/{doc}/
│   └── ner/{doc}/
├── training/                      # Training pipeline
│   ├── config/
│   ├── trainer.py
│   └── train.py
├── models/                        # Trained model output
├── inference/                     # Inference pipeline
│   ├── pipeline.py
│   ├── schema_mapper.py
│   └── postprocess.py
├── huggingface/                   # Publish to HuggingFace Hub
│   ├── push_model.py
│   ├── push_dataset.py
│   └── model_card.md
├── sdk/vietnerm/                  # Python SDK package
│   ├── __init__.py
│   ├── ner.py
│   └── utils.py
├── scripts/
│   ├── generate_data.sh
│   ├── train.sh
│   └── publish.sh
└── .github/workflows/
    └── train-and-publish.yml
```

---

## 🔄 Pipeline chi tiết

### Bước 1: Định nghĩa Document Type

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

### Bước 2: Sinh dữ liệu tổng hợp

```bash
# Sinh 10,000 mẫu cho CCCD
./scripts/generate_data.sh cccd 10000

# Sinh 50,000 mẫu cho Giấy ra viện
./scripts/generate_data.sh giay_ra_vien 50000
```

Pipeline tự động thực hiện:
1. Load template Jinja2
2. Sinh dữ liệu ngẫu nhiên (tên, ngày sinh, địa chỉ...)
3. Render template → text + character spans
4. Inject nhiễu OCR
5. Convert sang BIO NER format
6. Split train/test (80/10/10)

### Bước 3: Huấn luyện mô hình

```bash
# Train model cho CCCD
./scripts/train.sh cccd

# Train model với custom params
./scripts/train.sh giay_ra_vien --epochs 10 --batch_size 32
```

### Bước 4: Inference

```python
from inference.pipeline import NERPipeline
from inference.schema_mapper import SchemaMapper

pipeline = NERPipeline(model_path="models/phobert/cccd/")
raw_entities = pipeline.predict("Họ và tên: NGUYỄN VĂN A...")

mapper = SchemaMapper(doc_type="cccd")
result = mapper.map_entities(raw_entities)
```

---

## 📂 Templates & Registry

### Cấu trúc Templates

Mỗi loại giấy tờ có thư mục riêng với `schema.yaml` và các file template:

![Templates folder structure](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img1_templates.png)

*Thư mục `templates/` chứa 5 loại giấy tờ: CCCD, Giấy khai sinh, Giấy ra viện, GPLX, Đăng ký xe.*

### Registry — documents.yaml

File `registry/documents.yaml` là trung tâm đăng ký tất cả các loại giấy tờ:

![Registry documents.yaml](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img2_registry.png)

```yaml
documents:
  cccd:
    name: "Căn cước công dân"
    templates: templates/cccd
    schema: templates/cccd/schema.yaml
    generator: person

  giay_ra_vien:
    name: "Giấy ra viện"
    templates: templates/giay_ra_vien
    schema: templates/giay_ra_vien/schema.yaml
    generator: hospital

  vehicle_registration:
    name: "Đăng ký xe"
    templates: templates/vehicle_registration
    schema: templates/vehicle_registration/schema.yaml
    generator: vehicle

  gplx:
    name: "Giấy phép lái xe"
    templates: templates/gplx
    schema: templates/gplx/schema.yaml
    generator: gplx

  giay_khai_sinh:
    name: "Giấy khai sinh"
    templates: templates/giay_khai_sinh
    schema: templates/giay_khai_sinh/schema.yaml
    generator: giay_khai_sinh
```

### Thêm loại giấy tờ mới

1. Thêm entry vào `registry/documents.yaml`
2. Tạo `templates/{doc_type}/schema.yaml`
3. Tạo template files `templates/{doc_type}/template_*.txt`
4. Thêm generator vào `synthetic/generators/`
5. Thêm training config `training/config/{doc_type}.yaml`
6. Sinh dữ liệu → Train → Publish

Xem hướng dẫn chi tiết: [docs/adding_new_document.md](docs/adding_new_document.md)

---

## 🚀 Training trên Kaggle GPU

### GitHub Actions CI/CD

Dự án sử dụng GitHub Actions để tự động train trên **Kaggle GPU P100** và publish lên HuggingFace:

![GitHub Actions workflow in progress](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img3_github_actions.png)

*Workflow `train-and-publish.yml` gồm 2 bước: Train on Kaggle GPU → Training Report.*

### Kaggle Notebook — VietNerm Train Pipeline

Training được thực hiện trên Kaggle với GPU P100:

![Kaggle training logs - initialization](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img4_kaggle_logs.png)

**Log khởi động:**
```
Doc types to train (5):
  - cccd: Căn cước công dân
  - giay_ra_vien: Giấy ra viện
  - vehicle_registration: Đăng ký xe
  - gplx: Giấy phép lái xe
  - giay_khai_sinh: Giấy khai sinh

============================================================
[1/5] cccd — Căn cước công dân
============================================================
```

**Cấu hình môi trường:**

| Thông số | Giá trị |
|---|---|
| Accelerator | GPU P100 |
| Environment | Latest Container Image |
| Transformers | 4.51.3 (pinned, CVE-2025-32434 workaround) |
| Mixed Precision | fp16 |

---

## 📊 Kết quả Training

### Per-entity F1 Score — CCCD Model

![Training metrics and F1 scores](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img5_training_metrics.png)

Kết quả sau **epoch 1.0** trên tập validation:

| Entity | F1 Score |
|---|---|
| `full_name` | **0.9925** |
| `gender` | **0.9975** |
| `id_number` | **0.9975** |
| `nationality` | 0.9804 |
| `place_of_residence` | 0.9926 |
| `place_of_origin` | 0.9084 |
| `date_of_birth` | 0.7890 |
| `date_of_expiry` | 0.3436 |

**Eval F1 tổng thể:** `0.9101` sau epoch 1.0

**Training loss progression:**
```
Epoch 0.5: loss=1.4574
Epoch 1.0: loss=0.253
Epoch 1.5: loss=0.0809
Epoch 2.0: loss=0.0466
```

---

## 🤗 Publish lên HuggingFace

### Publish tự động

```bash
# Publish model và dataset
./scripts/publish.sh cccd <your_hf_username>
```

### Profile HuggingFace

![HuggingFace profile with 4 models and 4 datasets](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img6_huggingface_profile.png)

**Models đã publish:**
- `ngocthanhdoan/phobert-giay_ra_vien-ner`
- `ngocthanhdoan/phobert-cccd-ner`
- `ngocthanhdoan/phobert-gplx-ner`
- `ngocthanhdoan/phobert-vehicle_registration-ner`

**Datasets đã publish:**
- `ngocthanhdoan/vietnerm-giay_ra_vien-dataset`
- `ngocthanhdoan/vietnerm-cccd-dataset`
- `ngocthanhdoan/vietnerm-gplx-dataset`
- `ngocthanhdoan/vietnerm-vehicle_registration-dataset`

### Model Files — phobert-cccd-ner

![HuggingFace model files listing](https://raw.githubusercontent.com/Devhub-Solutions/VietNerm/refs/heads/main/docs/www/img7_huggingface_model.png)

Model `phobert-cccd-ner` (8.62 GB) bao gồm:

| File | Kích thước | Mô tả |
|---|---|---|
| `model.safetensors` | 538 MB | Weights chính |
| `bpe.codes` | 1.14 MB | BPE codes cho PhoBERT tokenizer |
| `vocab.txt` | 895 kB | Vocabulary |
| `label_map.json` | 706 Bytes | Mapping nhãn BIO |
| `config.json` | 1.43 kB | Cấu hình model |
| `checkpoint-100` ~ `checkpoint-500` | — | Checkpoints theo bước |

---

## 📦 SDK API Reference

### VietNerm — Class chính

```python
from vietnerm import VietNerm

# Tạo extractor cho CCCD
ner = VietNerm(doc_type="cccd")
ner = VietNerm(doc_type="cccd", model_path="path/to/model")
ner = VietNerm(doc_type="cccd", device="cuda")

# Trích xuất cơ bản
result = ner.extract("""
Họ và tên: NGUYỄN VĂN A
Ngày sinh: 01/01/1990
Giới tính: Nam
Quốc tịch: Việt Nam
""")
# Output: {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}

# Trích xuất kèm confidence
result = ner.extract_with_confidence(text)
# Output: {"name": {"value": "Nguyễn Văn A", "confidence": 0.99}, ...}

# Raw entities
raw = ner.extract_raw(text)
# Output: [{"entity": "B-full_name", "word": "NGUYỄN", "score": 0.99}, ...]
```

### Shortcut Classes

```python
from vietnerm import CCCDNer, GiayRaVienNer

# CCCD (Căn cước công dân)
cccd = CCCDNer()
result = cccd.extract("Số: 079203030140\nHọ và tên: NGUYỄN VĂN A")

# Giấy ra viện
grv = GiayRaVienNer()
result = grv.extract("Họ tên người bệnh: LÊ THỊ HẰNG\nChẩn đoán: Viêm phổi")
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

---

## 📄 Các loại giấy tờ hỗ trợ

| Loại giấy tờ | Doc Type | Entities |
|---|---|---|
| Căn cước công dân | `cccd` | Số CCCD, Họ tên, Ngày sinh, Giới tính, Quốc tịch, Quê quán, Nơi thường trú, Hạn sử dụng |
| Giấy ra viện | `giay_ra_vien` | Bệnh viện, Khoa, Mã y tế, Tên BN, Ngày sinh, Giới tính, Dân tộc, Nghề nghiệp, Địa chỉ, BHXH, Ngày vào/ra viện, Chẩn đoán, Phương pháp điều trị |
| Đăng ký xe | `vehicle_registration` | Tên chủ sở hữu, Biển số, Loại xe, Hãng, Số máy, Số khung |
| Giấy phép lái xe | `gplx` | Họ tên, Ngày sinh, Địa chỉ, Số GPLX, Hạng, Ngày cấp, Hạn dùng |
| Giấy khai sinh | `giay_khai_sinh` | Họ tên trẻ, Ngày sinh, Giới tính, Nơi sinh, Họ tên cha/mẹ |

---

## 🤝 Đóng góp

1. Fork repository
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Code theo conventions:
   - Type hints cho tất cả functions
   - Docstrings (Google style)
   - UTF-8 cho text tiếng Việt
   - `pathlib` cho đường dẫn file
4. Chạy lint: `flake8 . --max-line-length=120`
5. Tạo Pull Request

---

## 📜 License

```
Copyright (c) 2026 Devhub Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so.
```

---

*Được xây dựng với ❤️ bởi [Devhub Solutions](https://github.com/Devhub-Solutions) — Document AI cho Việt Nam*
