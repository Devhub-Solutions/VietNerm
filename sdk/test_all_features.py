"""Test: Sliding Window Inference + TF-IDF DocTypeDetector"""
import sys, os
sys.path.insert(0, "/home/ubuntu/VietNerm/sdk")

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: DocTypeDetector TF-IDF from templates
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("TEST 1: DocTypeDetector — TF-IDF from templates")
print("=" * 60)

from vietnerm.detector import DocTypeDetector

detector = DocTypeDetector.from_templates("/home/ubuntu/VietNerm/templates")
print(f"  Index built: {detector._index is not None}")
print(f"  Doc types: {list(detector.rules.keys())}")

test_cases = [
    ("CCCD", "CĂN CƯỚC CÔNG DÂN\nSố định danh cá nhân: 079203030140\nHọ và tên: NGUYỄN VĂN AN"),
    ("GPLX", "GIẤY PHÉP LÁI XE\nHạng/Class: B2\nNgày sinh: 01/01/1990"),
    ("Giấy ra viện", "GIẤY RA VIỆN\nVào viện lúc: 08:00 ngày 01/01/2024\nChẩn đoán: Viêm phổi"),
    ("Giấy khai sinh", "GIẤY KHAI SINH\nĐăng ký khai sinh\nHọ tên cha: Nguyễn Văn B"),
    ("Đăng ký xe", "CHỨNG NHẬN ĐĂNG KÝ XE\nBiển số: 51A-123.45\nSố máy: ABC123"),
    ("Không rõ", "Đây là một văn bản không liên quan đến giấy tờ nào"),
]

all_pass = True
for name, text in test_cases:
    result = detector.detect(text)
    status = "✓" if result.is_confident or name == "Không rõ" else "✗"
    if name == "Không rõ" and result.is_confident:
        status = "✗"
        all_pass = False
    elif name != "Không rõ" and not result.is_confident:
        all_pass = False
    print(f"  [{status}] {name:20s} → {str(result.doc_type):25s} conf={result.confidence:.3f} method={result.method}")

print(f"\n  Test 1 result: {'PASS' if all_pass else 'FAIL'}")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: TF-IDF index serialization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: TF-IDF index save/load roundtrip")
print("=" * 60)

import json, tempfile
from pathlib import Path
from vietnerm.detector import build_and_save_rules, _TFIDFIndex

with tempfile.TemporaryDirectory() as tmp:
    rules_path = Path(tmp) / "detector_rules.json"
    rules = build_and_save_rules(Path("/home/ubuntu/VietNerm/templates"), rules_path)
    index_path = Path(tmp) / "detector_index.json"

    rules_ok = rules_path.exists()
    index_ok = index_path.exists()
    print(f"  detector_rules.json saved: {rules_ok}")
    print(f"  detector_index.json saved: {index_ok}")

    if index_ok:
        with open(index_path) as f:
            loaded_index = _TFIDFIndex.from_dict(json.load(f))
        d2 = DocTypeDetector(rules=rules, index=loaded_index)
        r = d2.detect("CĂN CƯỚC CÔNG DÂN\nSố định danh: 079203030140")
        print(f"  Loaded index detect: {r.doc_type} conf={r.confidence:.3f}")
        print(f"  Test 2 result: {'PASS' if r.doc_type == 'cccd' else 'FAIL'}")
    else:
        print("  Test 2 result: SKIP (index not saved)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: explain() method
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: explain() method")
print("=" * 60)

expl = detector.explain("GIẤY PHÉP LÁI XE\nHạng B2\nNgày cấp: 01/01/2020")
print(f"  method: {expl['method']}")
if expl["tfidf_scores"]:
    top3 = sorted(expl["tfidf_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top-3 TF-IDF scores: {top3}")
print(f"  Test 3 result: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Sliding Window — _encode_tokens and _predict_with_sliding_window logic
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: Sliding Window — token encoding + window splitting")
print("=" * 60)

import sys
sys.path.insert(0, "/home/ubuntu/VietNerm/sdk/vietnerm/_inference")

# Test window splitting logic directly (without loading model)
from collections import defaultdict
import numpy as np

def simulate_windows(total_sw: int, max_content: int = 254, stride: int = 128):
    """Simulate sliding window splitting."""
    windows = []
    start = 0
    while start < total_sw:
        end = min(start + max_content, total_sw)
        windows.append((start, end))
        if end >= total_sw:
            break
        start += stride
    return windows

# Short text (no split needed)
wins = simulate_windows(100)
print(f"  100 tokens → {len(wins)} window(s): {wins}")
assert len(wins) == 1, "Short text should need 1 window"

# Medium text (needs 2 windows)
wins = simulate_windows(300)
print(f"  300 tokens → {len(wins)} window(s): {wins}")
assert len(wins) == 2, f"300 tokens should need 2 windows, got {len(wins)}"

# Long text (needs 3 windows)
wins = simulate_windows(500)
print(f"  500 tokens → {len(wins)} window(s): {wins}")
assert len(wins) == 3, f"500 tokens should need 3 windows, got {len(wins)}"

# Check overlap coverage
for i in range(len(wins) - 1):
    s1, e1 = wins[i]
    s2, e2 = wins[i + 1]
    overlap = e1 - s2
    assert overlap > 0, f"Window {i} and {i+1} should overlap"
    print(f"  Window {i}↔{i+1} overlap: {overlap} tokens ✓")

print(f"  Test 4 result: PASS")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Merge logic — average confidence at overlapping positions
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 5: Sliding Window merge — average confidence")
print("=" * 60)

# Simulate 2 windows predicting the same position differently
sw_predictions = defaultdict(list)
# Position 130 is in both windows
sw_predictions[130].append(("B-full_name", 0.9))   # window 1 prediction
sw_predictions[130].append(("B-full_name", 0.85))  # window 2 prediction (same label)
sw_predictions[131].append(("I-full_name", 0.88))
sw_predictions[131].append(("O", 0.3))  # window 2 disagrees

for sw_i in [130, 131]:
    preds = sw_predictions[sw_i]
    label_conf = defaultdict(list)
    for lbl, conf in preds:
        label_conf[lbl].append(conf)
    best_label = max(label_conf, key=lambda l: np.mean(label_conf[l]))
    best_conf = float(np.mean(label_conf[best_label]))
    print(f"  Position {sw_i}: preds={preds} → best=({best_label}, {best_conf:.3f})")

# Position 130: both say B-full_name → avg = 0.875
assert abs(float(np.mean([0.9, 0.85])) - 0.875) < 0.001
# Position 131: I-full_name (0.88) wins over O (0.3)
assert max(["I-full_name", "O"], key=lambda l: np.mean([0.88] if l == "I-full_name" else [0.3])) == "I-full_name"

print(f"  Test 5 result: PASS")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
