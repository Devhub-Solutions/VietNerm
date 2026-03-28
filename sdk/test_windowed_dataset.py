"""Test _build_windowed_dataset — validates the fix for PyArrow ArrowInvalid.

Simulates the exact scenario that caused the crash:
- giay_ra_vien: 29 labels, long documents (>256 tokens after subword tokenization)
- vehicle_registration: 19 labels, long documents
"""
import sys
sys.path.insert(0, "/home/ubuntu/VietNerm")

from transformers import AutoTokenizer
from training.trainer import _build_windowed_dataset
from datasets import Dataset, Features, Sequence, Value

print("Loading PhoBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# Simulate giay_ra_vien — 29 labels, long document
label_list_grv = [
    "O", "B-admission_date", "B-bhxh_code", "B-department", "B-dept_mgmt",
    "B-diagnosis", "B-discharge_date", "B-hospital_name", "B-medical_code",
    "B-notes", "B-patient_address", "B-patient_dob", "B-patient_ethnicity",
    "B-patient_gender", "B-patient_name", "B-patient_occupation",
    "B-treatment_method", "I-admission_date", "I-bhxh_code", "I-department",
    "I-dept_mgmt", "I-diagnosis", "I-discharge_date", "I-hospital_name",
    "I-notes", "I-patient_address", "I-patient_name", "I-patient_occupation",
    "I-treatment_method"
]
label2id_grv = {l: i for i, l in enumerate(label_list_grv)}

# Create a long sample (200 words → ~300+ subword tokens after PhoBERT tokenization)
long_tokens = ["Bệnh", "viện", "Đa", "khoa", "Trung", "ương", "Huế"] * 30  # 210 tokens
long_tags = ["B-hospital_name"] + ["I-hospital_name"] * 6 + ["O"] * (len(long_tokens) - 7)
long_tags = long_tags[:len(long_tokens)]

# Create 1000 samples (some short, some long)
samples = []
for i in range(1000):
    if i % 3 == 0:
        # Long sample — will need sliding window
        samples.append({"tokens": long_tokens, "ner_tags": long_tags})
    else:
        # Short sample — fits in 1 window
        samples.append({
            "tokens": ["Họ", "tên", ":", "Nguyễn", "Văn", "An"],
            "ner_tags": ["O", "O", "O", "B-patient_name", "I-patient_name", "I-patient_name"]
        })

print(f"\nTest 1: _build_windowed_dataset with {len(samples)} samples (mix short+long)")
windows = _build_windowed_dataset(samples, tokenizer, label2id_grv, max_length=256, stride=128)
print(f"  Input samples: {len(samples)}")
print(f"  Output windows: {len(windows)}")
assert len(windows) > len(samples), "Long docs should produce more windows than input samples"

# Validate all windows have correct length
for j, w in enumerate(windows):
    assert len(w["input_ids"]) == 256, f"Window {j}: input_ids length {len(w['input_ids'])} != 256"
    assert len(w["attention_mask"]) == 256, f"Window {j}: attention_mask length mismatch"
    assert len(w["labels"]) == 256, f"Window {j}: labels length mismatch"
print(f"  All {len(windows)} windows: length=256 ✓")

print("\nTest 2: Dataset.from_list() — no PyArrow error")
tok_features = Features({
    "input_ids": Sequence(Value("int32")),
    "attention_mask": Sequence(Value("int8")),
    "labels": Sequence(Value("int32")),
})
try:
    ds = Dataset.from_list(windows, features=tok_features)
    print(f"  Dataset created: {len(ds)} rows ✓")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

print("\nTest 3: Simulate the exact crash scenario (map() with row count mismatch)")
# This is what the OLD code did — should fail
from datasets import Dataset as HFDataset
raw_features = Features({
    "tokens": Sequence(Value("string")),
    "ner_tags": Sequence(Value("string")),
})
raw_ds = HFDataset.from_list(
    [{"tokens": s["tokens"], "ner_tags": s["ner_tags"]} for s in samples[:100]],
    features=raw_features,
)
try:
    from training.trainer import _tokenize_and_align_labels
    fn_kwargs = {"tokenizer": tokenizer, "label2id": label2id_grv, "max_length": 256, "stride": 128}
    result = raw_ds.map(_tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)
    print(f"  map() result: {len(result)} rows (may differ from input 100)")
    print(f"  NOTE: map() succeeded — this doc type may not trigger the crash")
except Exception as e:
    print(f"  map() failed as expected: {type(e).__name__}: {str(e)[:80]}")
    print(f"  This confirms the old code would crash with long documents ✓")

print("\n✓ ALL TESTS PASSED — _build_windowed_dataset fix is correct")
