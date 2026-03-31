# CI/CD: Kaggle GPU Training → HuggingFace Hub

## Tổng quan

Pipeline CI/CD tự động hóa toàn bộ quy trình training:

```
GitHub Push/Manual Trigger
    ↓
GitHub Actions (orchestrator)
    ↓
Kaggle API → Push kernel lên Kaggle (GPU T4/P100)
    ↓
Kaggle chạy: Clone repo → Generate data → Train PhoBERT → Evaluate
    ↓
GitHub Actions poll status → Download output
    ↓
Models đã được push lên HuggingFace Hub (từ trong Kaggle kernel)
    ↓
GitHub Actions hiển thị report
```

## Yêu cầu

### 1. Kaggle API Token

1. Vào [Kaggle Settings](https://www.kaggle.com/settings)
2. Scroll xuống phần **API** → Click **"Create New Token"**
3. File `kaggle.json` sẽ tự download, bên trong có:
   ```json
   {
     "username": "your-kaggle-username",
     "key": "your-kaggle-api-key"
   }
   ```

### 2. HuggingFace Token

1. Vào [HuggingFace Settings → Access Tokens](https://huggingface.co/settings/tokens)
2. Click **"New token"**
3. Chọn **Write** permission
4. Copy token

### 3. Setup GitHub Secrets

Vào repo GitHub → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**:

| Secret Name      | Giá trị                        | Bắt buộc |
|------------------|--------------------------------|-----------|
| `KAGGLE_USERNAME`| Kaggle username                | ✅        |
| `KAGGLE_KEY`     | Kaggle API key                 | ✅        |
| `HF_TOKEN`       | HuggingFace access token       | ✅        |
| `HF_USERNAME`    | HuggingFace username/org       | ❌ (default: `ngocthanhdoan`) |

## Cách sử dụng

### Chạy tự động

Pipeline tự động trigger khi push vào branch `main` và có thay đổi trong:
- `training/` — Training code
- `synthetic/` — Data generation
- `huggingface/` — HuggingFace push scripts
- `registry/` — Document types registry
- `templates/` — Jinja2 templates
- `kaggle/` — Kaggle kernel scripts

### Chạy thủ công (Manual Dispatch)

1. Vào repo → **Actions** → **"Train on Kaggle & Publish to HuggingFace"**
2. Click **"Run workflow"**
3. Tùy chọn:

| Parameter      | Default | Mô tả                                         |
|----------------|---------|------------------------------------------------|
| `doc_types`    | (all)   | Doc types cần train, comma-separated: `cccd,giay_ra_vien` |
| `dataset_size` | 2000    | Số samples synthetic data                      |
| `train_epochs` | 5       | Số epochs                                      |
| `batch_size`   | 16      | Batch size (Kaggle T4: 16-32)                  |
| `min_f1`       | 0.2     | F1 tối thiểu để publish model                  |
| `hf_private`   | false   | Tạo private repos trên HuggingFace             |

### Ví dụ sử dụng

**Train tất cả doc types:**
```
→ Run workflow (giữ default)
```

**Train chỉ CCCD với 10 epochs:**
```
doc_types: cccd
train_epochs: 10
```

**Train CCCD + Giấy ra viện, dataset lớn:**
```
doc_types: cccd,giay_ra_vien
dataset_size: 5000
train_epochs: 7
```

## Cấu trúc files

```
.github/workflows/
  └── train-and-publish.yml   ← GitHub Actions workflow

kaggle/
  ├── kernel-metadata.json    ← Kaggle kernel config (GPU, internet)
  └── vietnerm_train_kernel.py ← Training script chạy trên Kaggle
```

## Workflow chi tiết

### Job 1: `train-on-kaggle`

1. **Checkout code** — lấy code từ repo
2. **Install Kaggle CLI** — `pip install kaggle`
3. **Prepare kernel** — cập nhật `kernel-metadata.json` với username
4. **Inject config** — ghi env vars (HF_TOKEN, params) vào kernel script
5. **Push kernel** — `kaggle kernels push` để submit lên Kaggle
6. **Poll status** — kiểm tra mỗi 60s, max 2.5 giờ
7. **Download output** — `kaggle kernels output` để lấy results + models

### Job 2: `report`

- Parse `train_results.json`
- Tạo summary table trên GitHub Actions

## Lưu ý quan trọng

### Kaggle GPU Quota
- Kaggle cho **30 giờ GPU/tuần** miễn phí (T4)
- Mỗi session tối đa **12 giờ**
- Nếu hết quota, kernel sẽ chạy trên CPU (rất chậm)
- Kiểm tra quota: [kaggle.com/settings](https://www.kaggle.com/settings)

### HuggingFace
- Token phải có **Write** permission
- Model sẽ push lên `{HF_USERNAME}/phobert-{doc_type}-ner`
- Dataset sẽ push lên `{HF_USERNAME}/vietnerm-{doc_type}-dataset`

### Troubleshooting

| Vấn đề | Giải pháp |
|---------|-----------|
| Kernel status "error" | Kiểm tra log trong GitHub Actions artifacts |
| Timeout | Tăng `timeout-minutes` hoặc giảm `dataset_size` |
| F1 quá thấp | Tăng `dataset_size` và `train_epochs` |
| Kaggle quota hết | Chờ quota reset (hàng tuần) hoặc verify phone |
| HF push fail | Kiểm tra HF_TOKEN có write permission |

### Kaggle Secrets (Optional)

Ngoài inject env vars qua GitHub Actions, bạn cũng có thể setup secrets trực tiếp trên Kaggle:

1. Vào [kaggle.com](https://www.kaggle.com) → Your Profile → Settings
2. Hoặc trong notebook: Add-ons → Secrets
3. Thêm `HF_TOKEN` để dùng khi chạy kernel thủ công

## Security

- `HF_TOKEN` và `KAGGLE_KEY` được lưu trong GitHub Secrets (encrypted)
- Kernel script inject token lúc runtime, không commit vào code
- Kaggle kernel chạy ở chế độ **private**
