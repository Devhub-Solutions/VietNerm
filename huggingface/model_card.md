---
language: vi
tags:
  - ner
  - phobert
  - vietnamese
  - document-ai
  - ${doc_type}
  - synthetic-data
license: mit
base_model: ${base_model}
---

# VietNerm - ${doc_name} NER Model

PhoBERT-based Named Entity Recognition model for Vietnamese **${doc_name}** documents.

## ⚠️ DISCLAIMER: SYNTHETIC / MOCKUP DATA

> **Model này được train hoàn toàn trên dữ liệu giả lập (synthetic/mockup data), KHÔNG sử dụng dữ liệu cá nhân thật.**

- Tất cả dữ liệu training được **sinh tự động** bằng hệ thống template + generator
- **Không** sử dụng giấy tờ thật, thông tin cá nhân thật, hoặc dữ liệu thu thập từ người dùng
- Số định danh (ID, CCCD...) được sinh ngẫu nhiên, thiết kế để **không trùng** với dữ liệu thật
- Dữ liệu có inject nhiễu OCR (noise) để giả lập điều kiện thực tế
- Mục đích: **nghiên cứu AI, Document AI, OCR/NER pipeline**
- **Không** được sử dụng để giả mạo giấy tờ, tạo giấy tờ giả, lừa đảo hoặc gian lận

## Model Description

This model is fine-tuned from [`${base_model}`](https://huggingface.co/${base_model}) for token-level NER on Vietnamese administrative/medical documents. It extracts structured fields from OCR text output.

- **Base model**: ${base_model}
- **Task**: Token Classification (NER)
- **Language**: Vietnamese (vi)
- **Document type**: ${doc_name}
- **Number of labels**: ${num_labels}
- **Training data**: Synthetic/Mockup (not real personal data)

## Labels

${labels}

## Usage

### With VietNerm SDK

```python
from vietnerm import VietNerm

ner = VietNerm(doc_type="${doc_type}", model_path="${repo_id}")
result = ner.extract("your document text here")
print(result)
```

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("${repo_id}")
model = AutoModelForTokenClassification.from_pretrained("${repo_id}")

text = "your document text here"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

## Training

- **Dataset**: Synthetically generated (mockup data) with OCR noise simulation
- **Data source**: Auto-generated from Jinja2 templates + random generators (no real personal data)
- **Framework**: HuggingFace Transformers + Trainer API
- **Optimizer**: AdamW (lr=2e-5)
- **Epochs**: 5-7 (with early stopping)

## Ethical Use

This model is built for **research and development purposes only**:

- ✅ AI/NLP research
- ✅ Document AI development
- ✅ OCR/NER pipeline prototyping
- ✅ Educational purposes
- ❌ Forging documents
- ❌ Creating fake identity papers
- ❌ Fraud or deception

## About VietNerm

VietNerm is a Document AI Factory for Vietnamese documents. It provides a complete pipeline from template-based synthetic data generation to model training and deployment.

- **Repository**: [Devhub-Solutions/VietNerm](https://github.com/Devhub-Solutions/VietNerm)
- **License**: MIT — Copyright (c) 2026 Devhub Solutions
