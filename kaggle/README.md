# VietNerm — Train & Publish Pipeline (Kaggle Kernel)

## ⚠️ DISCLAIMER: DỮ LIỆU GIẢ LẬP (SYNTHETIC / MOCKUP DATA)

> **Toàn bộ dữ liệu trong pipeline này là dữ liệu giả lập (synthetic/mockup), được sinh tự động bằng hệ thống generator. KHÔNG sử dụng bất kỳ dữ liệu cá nhân thật nào.**

Cụ thể:
- **Không** sử dụng giấy tờ thật (CCCD, giấy ra viện, đăng ký xe...)
- **Không** sử dụng thông tin cá nhân thật (tên, số CCCD, ngày sinh...)
- **Không** thu thập dữ liệu từ người dùng
- Tất cả số định danh (ID) được sinh ngẫu nhiên, thiết kế để **không trùng** với dữ liệu thật
- Dữ liệu chỉ phục vụ mục đích **nghiên cứu AI và Document AI**

---

## Mô tả

Kernel này thực hiện toàn bộ pipeline training NER (Named Entity Recognition) cho tiếng Việt:

```
Clone repo → Install deps → Generate synthetic data → Train PhoBERT → Evaluate → Push to HuggingFace Hub
```

### Pipeline chi tiết

1. **Generate Data** — Sinh dữ liệu giả lập (synthetic) từ Jinja2 templates + noise engine (giả lập lỗi OCR)
2. **Train** — Fine-tune PhoBERT (`vinai/phobert-base`) cho NER token classification
3. **Evaluate** — Đánh giá F1 score, chỉ publish nếu đạt ngưỡng tối thiểu
4. **Publish** — Push model + dataset lên HuggingFace Hub

### Cấu hình

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `HF_USERNAME` | `ngocthanhdoan` | HuggingFace username/org |
| `DATASET_SIZE` | `2000` | Số lượng samples synthetic data |
| `TRAIN_EPOCHS` | `5` | Số epochs training |
| `BATCH_SIZE` | `16` | Batch size (T4: 16-32, A100: 32-64) |
| `MIN_F1` | `0.2` | F1 tối thiểu để cho phép publish |
| `DOC_TYPES` | (all) | Comma-separated doc types |

### Loại giấy tờ hỗ trợ

| Loại | Doc Type | Entities |
|------|----------|----------|
| Căn cước công dân | `cccd` | Số CCCD, Họ tên, Ngày sinh, Giới tính, Quốc tịch, Quê quán, Nơi thường trú |
| Giấy ra viện | `giay_ra_vien` | Bệnh viện, Khoa, Tên BN, Ngày sinh, Chẩn đoán, Ngày vào/ra viện... |
| Đăng ký xe | `vehicle_registration` | Tên chủ sở hữu, Biển số, Loại xe, Số máy, Số khung... |

### Yêu cầu

- **GPU**: T4 hoặc P100 (bật GPU trong Kaggle)
- **Internet**: Bật (để clone repo và push HuggingFace)
- **Kaggle Secrets**: `HF_TOKEN` (HuggingFace access token, write permission)

## Repository

- **GitHub**: [Devhub-Solutions/VietNerm](https://github.com/Devhub-Solutions/VietNerm)
- **HuggingFace Models**: [huggingface.co/ngocthanhdoan](https://huggingface.co/ngocthanhdoan)

## License

MIT — Copyright (c) 2026 Devhub Solutions
