# Hướng dẫn thêm loại giấy tờ mới (Document Type)

Khi người dùng yêu cầu thêm một loại giấy tờ mới vào hệ thống VietNerm, bạn (AI Agent) **bắt buộc** phải thực hiện đầy đủ các bước sau đây theo đúng thứ tự. Nếu thiếu một bước, toàn bộ pipeline (từ sinh dữ liệu đến inference) sẽ bị hỏng.

## Bước 1: Đăng ký vào Registry

Thêm một entry mới vào file `registry/documents.yaml`.

```yaml
documents:
  # ... các giấy tờ cũ ...
  new_doc_type:
    name: "Tên giấy tờ (VD: Giấy phép lái xe)"
    templates: templates/new_doc_type
    schema: templates/new_doc_type/schema.yaml
    generator: person  # hoặc hospital, vehicle tùy vào loại dữ liệu
```

## Bước 2: Định nghĩa Schema

Tạo file `templates/{new_doc_type}/schema.yaml`. File này định nghĩa các thực thể (entities) cần trích xuất.

```yaml
doc_type: new_doc_type
entities:
  - name: field_name_1
    type: person_name  # Các type hỗ trợ: id, person_name, date, gender, address, code...
  - name: field_name_2
    type: date
```

## Bước 3: Tạo Jinja2 Templates

Tạo ít nhất một file template tại `templates/{new_doc_type}/template_1.txt`. Template này giả lập cấu trúc văn bản của giấy tờ thật.

```text
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

GIẤY TỜ GÌ ĐÓ
Số: {{ field_name_1 }}
Họ và tên: {{ field_name_2 }}
```
**Lưu ý quan trọng**: Tên biến trong `{{ }}` **phải** khớp chính xác với `name` trong `schema.yaml`. Nên tạo nhiều template (template_2.txt, template_3.txt) với các biến thể format khác nhau để model học tốt hơn.

## Bước 4: Tạo hoặc Cập nhật Generator (Tùy chọn)

Nếu giấy tờ mới yêu cầu dữ liệu đặc thù không có trong các generator hiện tại (`person`, `hospital`, `vehicle`), bạn cần:
1. Tạo file mới `synthetic/generators/{new_generator}.py` kế thừa từ `BaseGenerator`.
2. Định nghĩa hàm `generate(self) -> Dict[str, str]` trả về dict chứa các field khớp với `schema.yaml`.
3. Import và thêm class mới vào biến `GENERATOR_MAP` trong file `synthetic/generators/__init__.py`.

## Bước 5: Cấu hình Huấn luyện

Tạo file cấu hình huấn luyện `training/config/{new_doc_type}.yaml`:

```yaml
doc_type: new_doc_type
model_name: vinai/phobert-base
max_length: 256  # Tăng lên 512 nếu giấy tờ dài
num_samples: 5000
epochs: 5
batch_size: 16
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_steps: 50
early_stopping_patience: 2
```

## Bước 6: Thêm Entity Mapping cho Inference

Mở file `inference/schema_mapper.py` và thêm mapping cho giấy tờ mới vào biến `DEFAULT_MAPPINGS`.
Bước này rất quan trọng để chuyển từ label BIO in hoa (do mô hình dự đoán) sang tên field chuẩn.

```python
class SchemaMapper:
    DEFAULT_MAPPINGS = {
        "cccd": { ... },
        "giay_ra_vien": { ... },
        "new_doc_type": {
            "FIELD_NAME_1": "field_name_1",
            "FIELD_NAME_2": "field_name_2",
        },
    }
```

## Bước 7: Kiểm tra (Smoke Test)

Trước khi báo cáo hoàn thành, hãy chạy thử script sinh dữ liệu với số lượng nhỏ để đảm bảo mọi module đã được liên kết đúng:

```bash
python synthetic/generate_dataset.py --doc new_doc_type --size 10
```

Nếu lệnh trên chạy thành công và tạo ra các file trong `datasets/ner/{new_doc_type}/`, việc thêm document type mới đã hoàn tất. CI/CD sẽ tự động lo phần training khi code được push lên nhánh `main`.
