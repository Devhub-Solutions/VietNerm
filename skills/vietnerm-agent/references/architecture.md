# VietNerm Architecture Reference

VietNerm là một pipeline AI hoàn chỉnh để sinh dữ liệu tổng hợp (synthetic data) và huấn luyện mô hình PhoBERT trích xuất thực thể (NER) từ các giấy tờ tiếng Việt.

## 1. Cấu trúc thư mục (Module Map)

| Module | Đường dẫn | Chức năng | Files quan trọng |
|---|---|---|---|
| **Registry** | `registry/` | Single source of truth cho doc types | `documents.yaml` |
| **Templates** | `templates/{doc_type}/` | Schema + Jinja2 templates sinh text | `schema.yaml`, `template_*.txt` |
| **Synthetic Data** | `synthetic/` | Sinh dữ liệu giả, render template, tạo nhiễu | `generate_dataset.py`, `generators/*.py`, `template_engine.py` |
| **Training** | `training/` | Config và script huấn luyện PhoBERT | `train.py`, `trainer.py`, `config/{doc_type}.yaml` |
| **SDK** | `sdk/vietnerm/` | Zero-code auto-discovery API | `ner.py` (VietNerm class duy nhất), `registry.py`, `detector.py` |
| **Inference** | `sdk/vietnerm/_inference/` | Pipeline NER, schema mapping, postprocess | `pipeline.py`, `schema_mapper.py`, `postprocess.py` |
| **CI/CD** | `.github/workflows/` | Auto train trên Kaggle, push HF Hub | `train-and-publish.yml` |
| **HuggingFace** | `huggingface/` | Scripts push model/dataset lên HF Hub | `push_model.py`, `push_dataset.py` |

## 2. Pipeline Sinh dữ liệu (Synthetic Data)

Luồng dữ liệu từ Template đến BIO Dataset:

1. **Generator** (`synthetic/generators/`): Sinh dict chứa giá trị ngẫu nhiên (vd: `{"full_name": "Nguyễn Văn A", ...}`).
2. **Template Engine** (`synthetic/template_engine.py`): Nhúng dict vào Jinja2 template, theo dõi character spans.
3. **Noise Engine** (`synthetic/noise_engine.py`): Thêm nhiễu OCR (sai chính tả, mất dấu) nhưng giữ spans.
4. **BIO Converter** (`synthetic/generate_dataset.py`): Chuyển character spans thành token-level BIO tags.
5. **Output**: `train.json`, `test.json`, `labels.json` trong `datasets/ner/{doc_type}/`.

## 3. Pipeline Huấn luyện (Training)

1. Load dataset từ `datasets/ner/{doc_type}/`.
2. Khởi tạo tokenizer + model `vinai/phobert-base`.
3. Re-tokenize + align BIO labels (subword đầu nhận label, các subword sau = `-100`).
4. Train với config `training/config/{doc_type}.yaml` (metric: F1-score).
5. Lưu model + tokenizer + config + `label_map.json` vào `models/phobert/{doc_type}/`.

## 4. Pipeline Suy luận (Inference & SDK)

Luồng xử lý khi gọi `VietNerm("doc_type").extract(text)`:

1. **Model Resolution**: `doc_type` → HF Hub repo `{hf_username}/phobert-{doc_type}-ner`
2. **NERPipeline** (`_inference/pipeline.py`):
   - Sliding window tokenization (max_length=256, PhoBERT limit)
   - Model predict → BIO tags
   - `merge_subtoken_predictions`: Gom subwords → words → entity spans
3. **Postprocess** (`_inference/postprocess.py`):
   - `clean_entity_boundaries`: Xóa dấu câu thừa, chuẩn hóa tên (Title Case), giới tính
   - `filter_validated_entities`: Regex validators loại bỏ entity sai format
4. **SchemaMapper** (`_inference/schema_mapper.py`):
   - **Auto-derive** mapping từ model `id2label` config (KHÔNG dùng DEFAULT_MAPPINGS)
   - Convention: BIO label `B-full_name` → entity type `FULL_NAME` → field `full_name`
   - Trả về dict `{field_name: value}` cho user

## 5. Auto-Discovery Flow

Khi SDK cần extract một doc type mới:

```
VietNerm("new_doc")
  → resolve: ngocthanhdoan/phobert-new_doc-ner
  → download model from HF Hub
  → read config.json → id2label: {0: "O", 1: "B-field_a", 2: "I-field_a", ...}
  → SchemaMapper auto-build: {"FIELD_A": "field_a", "FIELD_B": "field_b", ...}
  → extract(text) → {"field_a": "value", "field_b": "value", ...}
```

Không cần code mới. Chỉ cần model tồn tại trên HF Hub.

## 6. Model Registry & Discovery

SDK cung cấp 3 cách discover models:

1. **HF Hub scan**: `VietNerm.available_models()` — scan repos matching `phobert-*-ner`
2. **Local registry**: `ModelRegistry(local_registry_path=...)` — đọc `documents.yaml`
3. **Cache**: `~/.cache/vietnerm/model_registry.json` — TTL 1 giờ

## 7. CI/CD (Kaggle Training)

- GitHub Actions + Kaggle GPU (miễn phí)
- Workflow: `.github/workflows/train-and-publish.yml`
- Kernel: `kaggle/vietnerm_train_kernel.py`
- Auto: Sinh data → Train → Evaluate F1 → Push model + dataset lên HF Hub
- `push_model.py`: Dynamic doc_names từ registry, ignore checkpoints
- `push_dataset.py`: Dynamic doc_names từ registry
- Môi trường: Python 3.12, CUDA 12.1

## 8. Doc Type Detection

`DocTypeDetector` dùng TF-IDF + FAISS cosine similarity:

1. Build TF-IDF vectors từ template files
2. Query text → cosine similarity với mỗi doc type vector
3. Threshold mặc định: 0.25 (confidence), margin: 1.03 (best/second ratio)
4. Fallback: keyword scoring nếu không có FAISS index
