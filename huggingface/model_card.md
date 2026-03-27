---
language: vi
tags:
  - ner
  - phobert
  - vietnamese
  - document-ai
  - ${doc_type}
license: mit
base_model: ${base_model}
---

# VietNerm - ${doc_name} NER Model

PhoBERT-based Named Entity Recognition model for Vietnamese **${doc_name}** documents.

## Model Description

This model is fine-tuned from [`${base_model}`](https://huggingface.co/${base_model}) for token-level NER
on Vietnamese administrative/medical documents. It extracts structured fields from OCR text output.

- **Base model**: ${base_model}
- **Task**: Token Classification (NER)
- **Language**: Vietnamese (vi)
- **Document type**: ${doc_name}
- **Number of labels**: ${num_labels}

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

- **Dataset**: Synthetically generated with OCR noise simulation
- **Framework**: HuggingFace Transformers + Trainer API
- **Optimizer**: AdamW (lr=2e-5)
- **Epochs**: 5-7 (with early stopping)

## About VietNerm

VietNerm is a Document AI Factory for Vietnamese documents. It provides a complete pipeline
from template-based synthetic data generation to model training and deployment.

Learn more: [VietNerm Repository](https://github.com/vietnerm/vietnerm)
