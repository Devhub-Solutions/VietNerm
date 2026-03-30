# VietNerm 0.2.8 test log

Ngày chạy: 2026-03-30 (UTC)

## Cài đặt

```bash
pip install vietnerm==0.2.8
pip install vncv
apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0
```

## Script test đã chạy

```python
from vncv.ocr import extract_text
from vietnerm import DocTypeDetector, VietNerm, DownloadConfig

image_path = '/tmp/test_cccd.jpg'
text = extract_text(image_path)
full_text = "\n".join(text)

print('-- OCR RAW TEXT')
print(full_text)
print('--')

detection = DocTypeDetector().detect(full_text)
print(f'Detected document type: {detection.doc_type}')

config = DownloadConfig(disable_ssl_verify=True)
ner = VietNerm(download_config=config)
result = ner.extract(doc_type=detection.doc_type, text=full_text)
print('Extracted NER result:')
for key, value in result.items():
    print(f'{key}: {value}')
```

## Kết quả chính

- Import thành công `vncv.ocr` và `vietnerm`.
- OCR + detect + NER chạy thành công với ảnh test tạo tạm trong container.
- `DocTypeDetector` nhận diện `cccd`.
- Có trích xuất được `id_number` và `nationality`.

## Lưu ý

- Tên package đúng là `vietnerm==0.2.8` (không phải `vietnerm-0.2.8`).
- Trên Linux cần thêm `libGL.so.1` (gói `libgl1`) để import OpenCV từ `vncv`.
