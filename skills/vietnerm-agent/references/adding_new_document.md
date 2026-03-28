# Hướng dẫn thêm loại giấy tờ mới

Hướng dẫn từng bước để thêm một loại giấy tờ mới vào VietNerm pipeline.

## Ví dụ: Thêm "Giấy phép lái xe" (GPLX)

### Bước 1: Đăng ký Document Type

Thêm entry vào `registry/documents.yaml`:

```yaml
documents:
  # ... existing entries ...

  gplx:
    name: "Giấy phép lái xe"
    templates: templates/gplx
    schema: templates/gplx/schema.yaml
    generator: person
```

### Bước 2: Tạo Schema

Tạo file `templates/gplx/schema.yaml`:

```yaml
doc_type: gplx
entities:
  - name: full_name
    type: person_name
  - name: date_of_birth
    type: date
  - name: license_number
    type: id
  - name: license_class
    type: category
  - name: issue_date
    type: date
  - name: expiry_date
    type: date
  - name: place_of_issue
    type: address
  - name: nationality
    type: nationality
```

Mỗi entity cần:
- `name`: Tên field trong output dict
- `type`: Kiểu dữ liệu (dùng để chọn generator)

### Bước 3: Tạo Templates

Tạo ít nhất 1 template file `templates/gplx/template_1.txt`:

```
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

GIẤY PHÉP LÁI XE

Số: {{ license_number }}
Họ và tên: {{ full_name }}
Ngày sinh: {{ date_of_birth }}
Quốc tịch: {{ nationality }}
Hạng: {{ license_class }}
Ngày cấp: {{ issue_date }}
Có giá trị đến: {{ expiry_date }}
Nơi cấp: {{ place_of_issue }}
```

Lưu ý:
- Sử dụng Jinja2 placeholder `{{ field_name }}`
- Field names phải khớp với entity names trong schema
- Tạo nhiều template để tăng đa dạng (template_2.txt, template_3.txt)
- Có thể thêm biến thể label (tiếng Anh, viết tắt, v.v.)

### Bước 4: Tạo Data Generator

Nếu doc type dùng generator có sẵn (`person`, `hospital`, `vehicle`), bỏ qua bước này.

Nếu cần generator mới, tạo `synthetic/generators/gplx.py`:

```python
"""Generator cho dữ liệu Giấy phép lái xe."""

import random
from datetime import datetime, timedelta
from typing import Dict

from .base import BaseGenerator


class GPLXGenerator(BaseGenerator):
    """Sinh dữ liệu ngẫu nhiên cho GPLX."""

    LICENSE_CLASSES = ["A1", "A2", "B1", "B2", "C", "D", "E", "F"]

    ISSUE_PLACES = [
        "Sở GTVT Hà Nội",
        "Sở GTVT TP. Hồ Chí Minh",
        "Sở GTVT Đà Nẵng",
        # ... thêm tỉnh thành
    ]

    def generate(self) -> Dict[str, str]:
        issue_date = datetime(2015, 1, 1) + timedelta(
            days=random.randint(0, 3000)
        )
        expiry_date = issue_date + timedelta(days=3650)

        return {
            "full_name": self.random_name(),
            "date_of_birth": self.random_date(),
            "license_number": self.random_id(12),
            "license_class": random.choice(self.LICENSE_CLASSES),
            "issue_date": issue_date.strftime("%d/%m/%Y"),
            "expiry_date": expiry_date.strftime("%d/%m/%Y"),
            "place_of_issue": random.choice(self.ISSUE_PLACES),
            "nationality": "Việt Nam",
        }
```

Sau đó đăng ký generator trong `synthetic/generators/__init__.py`.

### Bước 5: Tạo Training Config

Tạo `training/config/gplx.yaml`:

```yaml
doc_type: gplx
model_name: vinai/phobert-base
max_length: 256
num_samples: 5000
epochs: 5
batch_size: 16
learning_rate: 2.0e-5
weight_decay: 0.01
warmup_steps: 50
early_stopping_patience: 2
```

### Bước 6: Thêm Entity Mapping

Thêm mapping vào `inference/schema_mapper.py`:

```python
# Trong SchemaMapper.DEFAULT_MAPPINGS
"gplx": {
    "FULL_NAME": "full_name",
    "DOB": "date_of_birth",
    "LICENSE_NUMBER": "license_number",
    "LICENSE_CLASS": "license_class",
    "ISSUE_DATE": "issue_date",
    "DATE_OF_EXPIRY": "expiry_date",
    "PLACE_OF_ISSUE": "place_of_issue",
    "NATIONALITY": "nationality",
},
```

### Bước 7: Sinh dữ liệu và Train

```bash
# Sinh dataset
./scripts/generate_data.sh gplx 10000

# Train model
./scripts/train.sh gplx

# Publish
./scripts/publish.sh gplx username
```

### Bước 8: Test với SDK

```python
from vietnerm import VietNerm

ner = VietNerm(doc_type="gplx")
result = ner.extract("""
GIẤY PHÉP LÁI XE
Số: 123456789012
Họ và tên: NGUYỄN VĂN A
Ngày sinh: 01/01/1990
Hạng: B2
""")
print(result)
# {"full_name": "Nguyễn Văn A", "license_number": "123456789012", ...}
```

## Checklist

- [ ] `registry/documents.yaml` - Entry mới
- [ ] `templates/{doc_type}/schema.yaml` - Schema
- [ ] `templates/{doc_type}/template_*.txt` - Ít nhất 1 template
- [ ] `synthetic/generators/` - Generator (nếu cần mới)
- [ ] `training/config/{doc_type}.yaml` - Training config
- [ ] `inference/schema_mapper.py` - Entity mapping
- [ ] Test sinh dữ liệu: `--size 10` chạy thành công
- [ ] Test training: Model train được
- [ ] Test inference: Extract entities đúng

## Mẹo

1. **Tạo nhiều template**: Càng nhiều template → model càng robust
2. **Đa dạng label text**: Thêm nhiều cách viết cho cùng 1 field (vd: "Họ tên:", "Full name:", "Họ và tên:")
3. **OCR noise thực tế**: Tham khảo dữ liệu OCR thực để calibrate noise engine
4. **Real data**: Nếu có dữ liệu thật, dùng `load_real_data()` để mix vào training (weight x3)
5. **Validators**: Thêm validator cho entity types mới trong `inference/postprocess.py`
