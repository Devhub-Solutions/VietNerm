# Kiến trúc VietNerm

## Tổng quan

VietNerm là một Document AI Factory - hệ thống pipeline tự động cho việc trích xuất thực thể (NER) từ giấy tờ tiếng Việt. Kiến trúc được thiết kế để thêm loại giấy tờ mới mà **không cần sửa code**.

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VietNerm Pipeline                            │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────┐ │
│  │ Registry │──▶│   Template   │──▶│ Synthetic  │──▶│  Dataset   │ │
│  │ (YAML)   │   │   Engine    │   │ Generator  │   │  Builder   │ │
│  └──────────┘   └──────────────┘   └───────────┘   └─────┬──────┘ │
│                                                           │        │
│  ┌──────────┐   ┌──────────────┐   ┌───────────┐        │        │
│  │   SDK    │◀──│  Inference   │◀──│ Training   │◀───────┘        │
│  │ Package  │   │  Pipeline    │   │ Pipeline   │                  │
│  └──────────┘   └──────────────┘   └───────────┘                  │
│       │                                    │                       │
│       │         ┌──────────────┐           │                       │
│       └────────▶│ HuggingFace  │◀──────────┘                      │
│                 │    Hub       │                                    │
│                 └──────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Các module chính

### 1. Registry (`registry/`)

Registry quản lý metadata cho tất cả loại giấy tờ. File `documents.yaml` chứa thông tin về:
- Tên hiển thị
- Đường dẫn template
- Schema file
- Generator type

Khi thêm loại giấy tờ mới, chỉ cần thêm entry vào file YAML này.

### 2. Templates (`templates/`)

Mỗi loại giấy tờ có thư mục riêng chứa:
- `schema.yaml`: Định nghĩa các entity cần extract
- `template_*.txt`: Jinja2 templates cho document layout

Templates sử dụng Jinja2 placeholders (`{{ field_name }}`) để đánh dấu vị trí entity.

### 3. Synthetic Data Generation (`synthetic/`)

#### Template Engine (`template_engine.py`)
- Load và render Jinja2 templates
- Track character offset cho mỗi entity value
- Output: `{"text": "...", "entities": [{"label": "name", "start": 12, "end": 24}]}`

#### Noise Engine (`noise_engine.py`)
- Giả lập lỗi OCR thực tế:
  - Thay thế ký tự (0↔o, l↔1)
  - Xóa/thêm ký tự
  - Lỗi dấu tiếng Việt
  - Random noise tokens

#### Generators (`generators/`)
- `person.py`: Tên, ngày sinh, CCCD, giới tính, quốc tịch, địa chỉ
- `hospital.py`: Bệnh viện, khoa, bệnh nhân, chẩn đoán, điều trị
- `vehicle.py`: Chủ sở hữu, biển số, loại xe, số máy, số khung
- `address.py`: Địa chỉ Việt Nam (tỉnh, huyện, xã, đường)

### 4. Dataset Builder (trong `generate_dataset.py`)

Quy trình:
1. Sinh raw records → lưu vào `datasets/raw/{doc}/`
2. Convert sang BIO NER format → lưu vào `datasets/ner/{doc}/`
3. Split train/valid/test (80/10/10)
4. Generate `labels.json`

#### BIO Tagging Strategy

VietNerm sử dụng chiến lược LABEL+VALUE tagging:
- `B-{ENTITY}_LABEL`: Token đầu tiên của nhãn (vd: "Họ và tên:")
- `I-{ENTITY}_LABEL`: Token tiếp theo của nhãn
- `B-{ENTITY}_VALUE`: Token đầu tiên của giá trị thực
- `I-{ENTITY}_VALUE`: Token tiếp theo của giá trị
- `O`: Token không thuộc entity nào

Việc tag cả LABEL giúp model học ngữ cảnh tốt hơn, không nhầm nhãn thành giá trị. Khi predict, chỉ lấy VALUE.

### 5. Training Pipeline (`training/`)

- **Base model**: `vinai/phobert-base` (PhoBERT)
- **Task**: Token Classification (NER)
- **Framework**: HuggingFace Transformers Trainer API
- **Hyperparameters** (mặc định):
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 5-7 (tùy doc type)
  - Early stopping: patience=2
  - FP16 (nếu CUDA khả dụng)

Config per doc type trong `training/config/{doc}.yaml` cho phép tùy chỉnh.

### 6. Inference Pipeline (`inference/`)

#### NERPipeline (`pipeline.py`)
- Load model từ local hoặc HuggingFace Hub
- Whitespace tokenization → PhoBERT subword tokenization
- Forward pass → BIO predictions với confidence scores
- Map subword predictions về word level

#### PostProcessor (`postprocess.py`)
- Merge B-/I- tags thành entity spans
- Clean entity boundaries
- Normalize values (gender, dates, names)
- Validate entities theo từng type
- Deduplication (giữ lần xuất hiện đầu tiên)

#### SchemaMapper (`schema_mapper.py`)
- Load schema.yaml cho doc type
- Map entity types → output field names
- Output structured dict

### 7. SDK (`sdk/vietnerm/`)

Package Python cài đặt được (`pip install -e ./sdk/`):
- `VietNerm`: Class chính, hỗ trợ mọi doc type
- `CCCDNer`: Shortcut cho CCCD
- `GiayRaVienNer`: Shortcut cho Giấy ra viện
- Auto-detect model path (local → HuggingFace Hub)

### 8. HuggingFace Integration (`huggingface/`)

- `push_model.py`: Upload trained model + auto-generated model card
- `push_dataset.py`: Upload NER dataset + dataset card
- `model_card.md`: Template cho model card

## Data Flow chi tiết

```
schema.yaml + template.txt
        │
        ▼
  Template Engine ──▶ {"text": "...", "entities": [...]}
        │
        ▼
  Noise Engine ──▶ {"text": "noisy...", "entities": [...]}
        │
        ▼
  BIO Converter ──▶ {"tokens": [...], "ner_tags": ["O", "B-NAME_VALUE", ...]}
        │
        ▼
  Train/Test Split ──▶ train.json, test.json, labels.json
        │
        ▼
  PhoBERT Trainer ──▶ models/phobert/{doc}/
        │
        ▼
  NERPipeline ──▶ raw entities
        │
        ▼
  SchemaMapper ──▶ {"name": "...", "dob": "..."}
```

## Nguyên tắc thiết kế

1. **Config-driven**: Thêm giấy tờ mới qua YAML, không sửa code
2. **Modular**: Mỗi module độc lập, dễ test riêng
3. **Reproducible**: Dataset có version, training có config file
4. **Production-ready**: Validation, error handling, confidence scores
5. **UTF-8 native**: Xử lý tiếng Việt đúng chuẩn Unicode NFC
