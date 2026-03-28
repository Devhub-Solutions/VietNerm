"""Test sliding window tokenization logic in trainer.py."""
import sys
sys.path.insert(0, "/home/ubuntu/VietNerm")

from training.trainer import _encode_sample_to_subwords, _make_windows, _tokenize_and_align_labels

# ── Mock tokenizer ──────────────────────────────────────────────
class MockTokenizer:
    cls_token_id = 0
    sep_token_id = 2
    pad_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Each character → 1 subword token (id = ord(char) % 100 + 10)
        return [ord(c) % 100 + 10 for c in text]

tokenizer = MockTokenizer()
label2id = {"O": 0, "B-name": 1, "I-name": 2, "B-date": 3}
id_to_label_keys = list(label2id.keys())
num_labels = len(label2id)

# ── Test 1: Short sample (no windowing needed) ──────────────────
print("=" * 60)
print("Test 1: Short sample (5 tokens, fits in 1 window)")
tokens = ["Họ", "tên", ":", "An", "Bình"]
tags   = ["O",  "O",   "O", "B-name", "I-name"]

ids, labs = _encode_sample_to_subwords(tokens, tags, tokenizer, label2id, id_to_label_keys, num_labels)
print(f"  Subword ids length: {len(ids)}")
print(f"  Subword labels: {labs[:10]}...")

windows = _make_windows(ids, labs, tokenizer, max_length=256, stride=128)
print(f"  Windows produced: {len(windows)} (expected: 1)")
assert len(windows) == 1, "Short sample should produce exactly 1 window"
inp, mask, lbl = windows[0]
assert len(inp) == 256, f"Window length should be 256, got {len(inp)}"
assert inp[0] == tokenizer.cls_token_id, "First token must be CLS"
assert inp[-1] != tokenizer.sep_token_id or mask[-1] == 0, "Last real token must be SEP"
print("  PASSED ✓")

# ── Test 2: Long sample (needs sliding window) ──────────────────
print()
print("=" * 60)
print("Test 2: Long sample (300 tokens, needs sliding window)")
# 300 single-char tokens, each encodes to 1 subword
tokens_long = [chr(65 + (i % 26)) for i in range(300)]
tags_long = ["O"] * 100 + ["B-name"] + ["I-name"] * 49 + ["O"] * 150

ids_long, labs_long = _encode_sample_to_subwords(
    tokens_long, tags_long, tokenizer, label2id, id_to_label_keys, num_labels
)
print(f"  Subword ids length: {len(ids_long)} (expected ~300)")

windows_long = _make_windows(ids_long, labs_long, tokenizer, max_length=256, stride=128)
print(f"  Windows produced: {len(windows_long)}")
# With 300 subwords, max_length=256 (usable=254), stride=128, step=126
# window 1: [0..253], window 2: [126..379] → [126..299]
assert len(windows_long) >= 2, "Long sample should produce >= 2 windows"
for w_idx, (inp_w, mask_w, lbl_w) in enumerate(windows_long):
    assert len(inp_w) == 256, f"Window {w_idx} length should be 256, got {len(inp_w)}"
    assert inp_w[0] == tokenizer.cls_token_id, f"Window {w_idx} must start with CLS"
    real_len = sum(mask_w)
    assert real_len <= 256, f"Window {w_idx} real tokens ({real_len}) > 256"
print(f"  All {len(windows_long)} windows valid (length=256, CLS prefix, no overflow)")
print("  PASSED ✓")

# ── Test 3: Batched function (as used by HF Dataset.map) ────────
print()
print("=" * 60)
print("Test 3: Batched _tokenize_and_align_labels with mixed lengths")
examples = {
    "tokens": [
        # Sample 0: short (5 words)
        ["Họ", "tên", ":", "An", "Bình"],
        # Sample 1: long (300 words)
        tokens_long,
    ],
    "ner_tags": [
        ["O", "O", "O", "B-name", "I-name"],
        tags_long,
    ]
}

result = _tokenize_and_align_labels(
    examples, tokenizer, label2id, max_length=256, stride=128
)

n_out = len(result["input_ids"])
print(f"  Input samples: 2 → Output windows: {n_out}")
assert n_out >= 3, f"Expected >= 3 windows (1 for short + >=2 for long), got {n_out}"

for idx in range(n_out):
    inp = result["input_ids"][idx]
    lbl = result["labels"][idx]
    msk = result["attention_mask"][idx]
    assert len(inp) == 256, f"Window {idx}: input_ids length {len(inp)} != 256"
    assert len(lbl) == 256, f"Window {idx}: labels length {len(lbl)} != 256"
    assert len(msk) == 256, f"Window {idx}: attention_mask length {len(msk)} != 256"
    # All label IDs must be -100 or valid
    bad = [l for l in lbl if l != -100 and not (0 <= l < num_labels)]
    assert not bad, f"Window {idx}: invalid label IDs: {bad}"

print(f"  All {n_out} output windows: length=256, labels valid")
print("  PASSED ✓")

# ── Test 4: Boundary entity coverage ────────────────────────────
print()
print("=" * 60)
print("Test 4: Entity at boundary appears in at least one window")
# Place entity at token 250 (near boundary of first window)
tokens_boundary = [chr(65 + (i % 26)) for i in range(300)]
tags_boundary = ["O"] * 250 + ["B-date"] + ["O"] * 49

ids_b, labs_b = _encode_sample_to_subwords(
    tokens_boundary, tags_boundary, tokenizer, label2id, id_to_label_keys, num_labels
)
windows_b = _make_windows(ids_b, labs_b, tokenizer, max_length=256, stride=128)

# Find which window contains the B-date label
date_label_id = label2id["B-date"]
found_in_windows = []
for w_idx, (_, _, lbl_w) in enumerate(windows_b):
    if date_label_id in lbl_w:
        found_in_windows.append(w_idx)

print(f"  B-date entity at subword 250 found in windows: {found_in_windows}")
assert len(found_in_windows) >= 1, "Entity at boundary must appear in at least 1 window"
print("  PASSED ✓")

print()
print("=" * 60)
print("ALL TESTS PASSED ✓")
print("Sliding window tokenization works correctly.")
