---
language: vi
tags:
  - ner
  - vietnamese
  - document-ai
  - ${doc_type}
  - synthetic-data
task_categories:
  - token-classification
size_categories:
  - 1K<n<10K
license: mit
---

# VietNerm — ${doc_name} NER Dataset

Synthetic BIO-tagged NER dataset for Vietnamese **${doc_name}** document entity extraction.

## ⚠️ DISCLAIMER: SYNTHETIC / MOCKUP DATA

> **Dataset này được sinh hoàn toàn tự động từ template (synthetic/mockup data), KHÔNG chứa dữ liệu cá nhân thật.**

- Tất cả dữ liệu được **sinh tự động** bằng hệ thống Jinja2 template + random generator
- **Không** sử dụng giấy tờ thật, thông tin cá nhân thật, hoặc dữ liệu thu thập từ người dùng
- Số định danh (ID, CCCD...) được sinh ngẫu nhiên, thiết kế để **không trùng** với dữ liệu thật
- Dữ liệu có inject nhiễu OCR (noise) để giả lập điều kiện thực tế
- Mục đích: **nghiên cứu AI, Document AI, OCR/NER pipeline**
- **Không** được sử dụng để giả mạo giấy tờ, tạo giấy tờ giả, lừa đảo hoặc gian lận

## Dataset Description

This dataset contains BIO-tagged token sequences for training NER models on Vietnamese
**${doc_name}** documents. Data is synthetically generated with OCR noise simulation for robustness.

### Dataset Statistics

| Split  | Samples |
|--------|---------|
| Train  | ${train_count} |
| Test   | ${test_count} |

### Labels

${labels_table}

## Format

Each sample is a JSON object with two fields:

| Field      | Type           | Description                         |
|------------|----------------|-------------------------------------|
| `tokens`   | `List[str]`    | Whitespace-tokenized words          |
| `ner_tags` | `List[str]`    | BIO label for each token            |

## Data Mockup Example

Below is a representative (synthetic) sample from the dataset:

```json
${mockup_example}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("${repo_id}")
train = dataset["train"]

# Access a sample
sample = train[0]
print(sample["tokens"])    # ['CĂN', 'CƯỚC', 'CÔNG', 'DÂN', ...]
print(sample["ner_tags"])  # ['O', 'O', 'O', 'O', ...]
```

## Training the NER Model

This dataset is used to train the companion model [`${hf_username}/phobert-${doc_type}-ner`](https://huggingface.co/${hf_username}/phobert-${doc_type}-ner).

```python
from vietnerm import VietNerm

ner = VietNerm(doc_type="${doc_type}", hf_username="${hf_username}")
result = ner.extract("your document OCR text here")
print(result)
```

## Ethical Use

This dataset is built for **research and development purposes only**:

- ✅ AI/NLP research
- ✅ Document AI development
- ✅ OCR/NER pipeline prototyping
- ✅ Educational purposes
- ❌ Forging documents
- ❌ Creating fake identity papers
- ❌ Fraud or deception

## About VietNerm

VietNerm is a Document AI Factory for Vietnamese documents. It provides a complete pipeline
from template-based synthetic data generation to model training and deployment.

- **Repository**: [Devhub-Solutions/VietNerm](https://github.com/Devhub-Solutions/VietNerm)
- **SDK**: `pip install vietnerm`
- **License**: MIT — Copyright (c) 2026 Devhub Solutions
