---
name: vietnerm-agent
description: Hướng dẫn dành cho AI Agent làm việc với dự án VietNerm (Document AI Factory cho giấy tờ Việt Nam). Kích hoạt khi người dùng yêu cầu đọc hiểu source code, sửa lỗi pipeline, thêm loại giấy tờ mới, hoặc giải thích luồng sinh dữ liệu và huấn luyện mô hình PhoBERT.
---

# VietNerm Agent Skill

Chào mừng bạn đến với dự án **VietNerm**. Đây là một hệ thống AI pipeline hoàn chỉnh dùng để trích xuất thực thể (NER) từ các loại giấy tờ tiếng Việt bằng cách sử dụng kiến trúc PhoBERT. Dự án này tự động hóa toàn bộ quy trình từ việc sinh dữ liệu giả (synthetic data) đến huấn luyện mô hình và public lên HuggingFace.

Skill này cung cấp cho bạn (AI Agent) những kiến thức về kiến trúc dự án và quy trình làm việc chuẩn để bạn có thể sửa lỗi hoặc mở rộng dự án một cách an toàn và chính xác.

## 1. Kiến trúc và Luồng dữ liệu (Architecture & Data Flow)

VietNerm được thiết kế theo hướng data-driven và config-driven. Thay vì viết code cứng cho từng loại giấy tờ, hệ thống sử dụng các file cấu hình và template.

Nếu bạn cần tìm hiểu cách dữ liệu chảy từ Template đến Dataset, cách mô hình được huấn luyện, hoặc cách Inference API hoạt động, hãy đọc file tham khảo sau:

**Đọc ngay**: `agent-skill/references/architecture.md` (trong repo VietNerm)

## 2. Quy trình thêm Document Type mới (Thường gặp nhất)

Một trong những task phổ biến nhất trong dự án này là thêm hỗ trợ cho một loại giấy tờ mới (ví dụ: Giấy phép lái xe, Hộ chiếu, Thẻ bảo hiểm y tế).

Quy trình này đòi hỏi phải cập nhật đồng bộ nhiều file ở nhiều thư mục khác nhau. Nếu bạn bỏ sót một bước, hệ thống sẽ báo lỗi.

**Đọc ngay hướng dẫn chi tiết**: `agent-skill/references/add_new_doc.md` (trong repo VietNerm)

Tóm tắt các bước bắt buộc:
1. Đăng ký vào `registry/documents.yaml`
2. Tạo `templates/{doc_type}/schema.yaml`
3. Tạo các `templates/{doc_type}/template_*.txt`
4. Cập nhật `synthetic/generators/` (nếu cần)
5. Tạo `training/config/{doc_type}.yaml`
6. Thêm mapping vào `inference/schema_mapper.py`

## 3. Các lưu ý quan trọng khi Debug (Troubleshooting)

Khi người dùng báo lỗi, hãy kiểm tra các nguyên nhân phổ biến sau:

### Lỗi Sinh dữ liệu (`generate_dataset.py`)
- **KeyError / Missing Field**: Thường do tên biến trong Jinja2 template (`{{ field_name }}`) không khớp chính xác với `name` được định nghĩa trong `schema.yaml`.
- **Unknown Generator**: Do quên đăng ký class generator mới vào biến `GENERATOR_MAP` trong `synthetic/generators/__init__.py`.

### Lỗi Inference / Kết quả rỗng
- **Label Mismatch**: Do mô hình dự đoán ra label in hoa (vd: `B-FULL_NAME`), nhưng trong `inference/schema_mapper.py` chưa định nghĩa mapping để chuyển `FULL_NAME` thành field name chuẩn.
- **Validation Filter**: File `inference/postprocess.py` có chứa các hàm regex validate. Nếu dữ liệu sinh ra (hoặc dữ liệu thật) không khớp regex (vd: ngày tháng sai format), entity đó sẽ bị filter bỏ và không xuất hiện trong kết quả cuối.

### Lỗi CI/CD (Kaggle Training)
- Dự án sử dụng Kaggle GPU thông qua GitHub Actions (`.github/workflows/train-and-publish.yml`).
- Kernel chạy trên Kaggle: `kaggle/vietnerm_train_kernel.py`. Môi trường Kaggle hiện tại sử dụng **Python 3.12 và CUDA 12.1**.
- **GPU Compatibility**: Kaggle có thể cấp **T4 (sm_75)** hoặc **P100 (sm_60)**. PyTorch 2.x+cu121 chỉ hỗ trợ sm_70+, nên **P100 sẽ crash** với lỗi `CUDA error: no kernel image for device`. Kernel đã được fix để tự động detect compute capability qua `nvidia-smi` và **fallback CPU** nếu sm < 7.0.
- **F1 null / `—` trong summary**: Xảy ra khi training crash trước khi eval → không có `trainer_state.json`. Kernel đã được fix để dùng `None` sentinel và chạy standalone evaluation làm fallback.
- Log lỗi của Kaggle có thể xem bằng cách dùng GitHub CLI: `gh run view <run-id> --log`.

## 4. Quy tắc Code (Coding Conventions)

Khi viết code mới cho dự án này, bạn phải tuân thủ:
- **Type Hints**: Bắt buộc sử dụng type hints cho tất cả các tham số và giá trị trả về của hàm (vd: `def extract(self, text: str) -> Dict[str, str]:`).
- **Docstrings**: Sử dụng Google-style docstrings cho các class và function public.
- **Pathlib**: Sử dụng thư viện `pathlib.Path` thay vì `os.path` cho các thao tác với đường dẫn file.
- **Tiếng Việt**: File text phải luôn dùng `encoding="utf-8"`.

## 5. Chạy Test

Sau khi sửa code hoặc thêm giấy tờ mới, luôn luôn chạy thử một lượng nhỏ dữ liệu để đảm bảo không có lỗi runtime:

```bash
python synthetic/generate_dataset.py --doc <doc_type> --size 10
```
