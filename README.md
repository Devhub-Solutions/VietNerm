# VietNerm : PhoBERT Pipeline to Hugging Face

Dự án này là một hệ thống hoàn chỉnh để xây dựng mô hình trích xuất thực thể (NER) cho các loại giấy tờ tiếng Việt (Giấy ra viện, CCCD) dựa trên kiến trúc PhoBERT. Hệ thống bao gồm quy trình từ sinh dữ liệu mẫu (synthetic data), tạo dataset, huấn luyện mô hình và tự động deploy lên Hugging Face Hub.

## 🌟 Tính năng chính

- **Synthetic Data Engine**: Tự động sinh hàng ngàn mẫu dữ liệu từ template văn bản với thông tin ngẫu nhiên (tên, ngày sinh, địa chỉ, bệnh lý...).
- **OCR Noise Simulation**: Giả lập các lỗi thường gặp khi quét OCR (nhầm lẫn số 0 và chữ o, mất ký tự...) để tăng độ bền vững cho mô hình.
- **PhoBERT NER Pipeline**: Sử dụng `vinai/phobert-base` để huấn luyện mô hình trích xuất thực thể với độ chính xác cao.
- **Hugging Face Integration**: Tích hợp sẵn script upload model và dataset lên Hugging Face Hub.
- **CI/CD Automation**: Tự động hóa quy trình deploy thông qua GitHub Actions.

## 📁 Cấu trúc dự án

```text
vietnerm/
├── .github/workflows/      # Cấu hình CI/CD (GitHub Actions)
├── template_engine/        # Module xử lý template và sinh dữ liệu
│   ├── config.py           # Danh sách dữ liệu mẫu (tên, bệnh viện, bệnh lý...)
│   ├── generators.py       # Logic sinh record ngẫu nhiên
│   └── noise.py            # Giả lập nhiễu OCR
├── dataset/                # Module tạo và quản lý dataset
│   └── builder.py          # Chuyển đổi record sang định dạng BIO cho NER
├── train/                  # Module huấn luyện mô hình
│   └── train_giaravien.py  # Script huấn luyện PhoBERT cho Giấy ra viện
├── inference/              # Module dự đoán (inference)
├── publish_to_hf.py        # Script upload model/dataset lên Hugging Face
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Tài liệu hướng dẫn
```

## 🛠️ Hướng dẫn sử dụng

### 1. Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 2. Quy trình huấn luyện và Deploy
1. **Sinh dữ liệu và huấn luyện**: Chạy script trong thư mục `train/` để bắt đầu quy trình.
2. **Cấu hình Secrets trên GitHub**:
   - `HF_TOKEN`: Hugging Face Write Token.
   - `HF_REPO_ID`: ID của model repo (ví dụ: `username/phobert-giaravien-ner`).
3. **Tự động Deploy**: Mỗi khi bạn push code hoặc cập nhật model trong thư mục `models/`, GitHub Actions sẽ tự động upload phiên bản mới nhất lên Hugging Face.

### 3. Upload thủ công
```bash
python publish_to_hf.py --path ./models/phobert_giaravien_ner --repo_id username/my-model --repo_type model
```

---
Dự án được tối ưu hóa cho các loại giấy tờ hành chính và y tế tại Việt Nam.
