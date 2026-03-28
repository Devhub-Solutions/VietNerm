# VietNerm Architecture Reference

VietNerm là một pipeline AI hoàn chỉnh để sinh dữ liệu tổng hợp (synthetic data) và huấn luyện mô hình PhoBERT trích xuất thực thể (NER) từ các giấy tờ tiếng Việt.

## 1. Cấu trúc thư mục (Module Map)

Khi cần sửa đổi hoặc debug, hãy tìm đến các module sau:

| Module | Đường dẫn | Chức năng | Files quan trọng |
|---|---|---|---|
| **Registry** | `registry/` | Nơi đăng ký các loại giấy tờ được hỗ trợ. | `documents.yaml` |
| **Templates** | `templates/{doc_type}/` | Chứa schema định nghĩa entity và các Jinja2 templates sinh text. | `schema.yaml`, `template_*.txt` |
| **Synthetic Data** | `synthetic/` | Logic sinh dữ liệu giả, render template và tạo nhiễu (noise). | `generate_dataset.py`, `generators/*.py`, `template_engine.py` |
| **Training** | `training/` | Config và script huấn luyện PhoBERT. | `train.py`, `trainer.py`, `config/{doc_type}.yaml` |
| **Inference** | `inference/` | Pipeline dự đoán NER, xử lý BIO tags thành entity. | `pipeline.py`, `schema_mapper.py`, `postprocess.py` |
| **SDK** | `sdk/vietnerm/` | API wrapper cho end-user. | `ner.py` |
| **CI/CD** | `.github/workflows/` | Tự động hóa train trên Kaggle và push lên HuggingFace. | `train-and-publish.yml` |

## 2. Pipeline Sinh dữ liệu (Synthetic Data)

Luồng đi của dữ liệu từ Template đến BIO Dataset:

1. **Generator** (`synthetic/generators/`): Sinh ra một dict chứa các giá trị ngẫu nhiên (vd: `{"full_name": "Nguyễn Văn A", ...}`).
2. **Template Engine** (`synthetic/template_engine.py`): Nhúng dict vào Jinja2 template (`templates/{doc_type}/template_*.txt`), theo dõi vị trí (character spans) của từng entity.
3. **Noise Engine** (`synthetic/noise_engine.py`): Thêm nhiễu OCR ngẫu nhiên vào text (sai chính tả, mất dấu) nhưng vẫn giữ nguyên character spans.
4. **BIO Converter** (`synthetic/generate_dataset.py`): Chuyển đổi character spans thành token-level BIO tags (`B-name`, `I-name`, `O`).
5. **Output**: Lưu thành `train.json`, `test.json`, `labels.json` trong `datasets/ner/{doc_type}/`.

## 3. Pipeline Huấn luyện (Training)

1. Load dataset từ `datasets/ner/{doc_type}/`.
2. Khởi tạo tokenizer và model `vinai/phobert-base`.
3. Re-tokenize text theo subword của PhoBERT và align lại BIO labels (chỉ gắn label cho subword đầu tiên của một từ, các subword sau đánh `-100` để ignore loss).
4. Train với cấu hình trong `training/config/{doc_type}.yaml` (mặc định: F1-score để chọn model tốt nhất).
5. Lưu model, tokenizer, config và `label_map.json` vào `models/phobert/{doc_type}/`.

## 4. Pipeline Suy luận (Inference & SDK)

Luồng xử lý khi người dùng gọi `VietNerm.extract(text)`:

1. **NERPipeline** (`inference/pipeline.py`):
   - Nhận text, tách thành các token (cách nhau bởi khoảng trắng).
   - Tokenize từng token thành subwords bằng PhoBERT tokenizer.
   - Chạy model predict ra list BIO tags.
   - `merge_subtoken_predictions`: Gom các subword tags lại thành word tags, sau đó gom các `B-` và `I-` liền kề thành một span entity.
2. **Postprocess** (`inference/postprocess.py`):
   - Lọc bỏ các label không chứa `_VALUE`.
   - `clean_entity_boundaries`: Xóa dấu câu thừa, chuẩn hóa tên riêng (Title Case), chuẩn hóa giới tính.
   - `filter_validated_entities`: Chạy regex validator để loại bỏ các entity sai định dạng (vd: ngày tháng không hợp lệ).
3. **SchemaMapper** (`inference/schema_mapper.py`):
   - Đọc `DEFAULT_MAPPINGS` để map từ BIO label (vd: `FULL_NAME`) sang schema field name (vd: `name`).
   - Trả về dict cuối cùng cho người dùng.

## 5. CI/CD (Kaggle Training)

Dự án dùng GitHub Actions kết hợp Kaggle GPU để train model miễn phí:
- File `train-and-publish.yml` đẩy code và config (qua env vars) lên Kaggle.
- Kernel `kaggle/vietnerm_train_kernel.py` chạy pipeline: Sinh dữ liệu → Train → Evaluate F1 → Push model/dataset lên HuggingFace.
- Nếu bạn cần sửa lỗi môi trường (vd: version thư viện), hãy kiểm tra phần cài đặt dependencies trong `vietnerm_train_kernel.py`. Môi trường Kaggle hiện tại dùng Python 3.12 và CUDA 12.1.
