"""
VietNerm — Kaggle Training Kernel
==================================
Script này chạy trên Kaggle với GPU (T4/P100).
Pipeline: Clone repo → Install deps → Generate data → Train → Evaluate → Push to HuggingFace Hub

Biến môi trường cần set (qua Kaggle Secrets hoặc env):
  - HF_TOKEN: HuggingFace access token (write)
  - HF_USERNAME: HuggingFace username/org (default: ngocthanhdoan)
  - DOC_TYPES: Comma-separated doc types (default: all from registry)
  - DATASET_SIZE: Number of samples to generate (default: 2000)
  - TRAIN_EPOCHS: Number of training epochs (default: 5)
  - BATCH_SIZE: Training batch size (default: 16)
  - MIN_F1: Minimum F1 to publish (default: 0.2)
  - GITHUB_REPO: GitHub repo URL (default: Devhub-Solutions/VietNerm)
  - GITHUB_BRANCH: Branch to clone (default: main)
"""

import os
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# ============================================================
# 1. CẤU HÌNH
# ============================================================

def get_env(key, default=None):
    """Lấy config từ Kaggle Secrets hoặc env vars."""
    try:
        from kaggle_secrets import UserSecretsClient
        client = UserSecretsClient()
        val = client.get_secret(key)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, default)


HF_TOKEN = get_env("HF_TOKEN", "")
HF_USERNAME = get_env("HF_USERNAME", "ngocthanhdoan")
DOC_TYPES_STR = get_env("DOC_TYPES", "")  # comma-separated, empty = all
DATASET_SIZE = int(get_env("DATASET_SIZE", "2000"))
TRAIN_EPOCHS = int(get_env("TRAIN_EPOCHS", "5"))
BATCH_SIZE = int(get_env("BATCH_SIZE", "16"))
MIN_F1 = float(get_env("MIN_F1", "0.2"))
GITHUB_REPO = get_env("GITHUB_REPO", "https://github.com/Devhub-Solutions/VietNerm.git")
GITHUB_BRANCH = get_env("GITHUB_BRANCH", "main")
HF_PRIVATE = get_env("HF_PRIVATE", "false").lower() == "true"

# ============================================================
# 2. SETUP MÔI TRƯỜNG
# ============================================================

print("=" * 60)
print("VietNerm — Kaggle Training Pipeline")
print("=" * 60)
print(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version}")
print(f"HF_TOKEN: {'SET (' + HF_TOKEN[:4] + '...)' if HF_TOKEN else 'NOT SET — models will NOT be pushed to HuggingFace!'}")
print(f"HF_USERNAME: {HF_USERNAME}")
print(f"DOC_TYPES: {DOC_TYPES_STR or 'all'}")
print(f"DATASET_SIZE: {DATASET_SIZE}")
print(f"TRAIN_EPOCHS: {TRAIN_EPOCHS}")
print(f"BATCH_SIZE: {BATCH_SIZE}")

# ── Kiểm tra GPU và CUDA compatibility ──────────────────────
# Kaggle có thể cấp T4 (sm_75) hoặc P100 (sm_60).
# PyTorch >= 2.0 chỉ hỗ trợ sm_70+, nên P100 KHÔNG tương thích.
# Giải pháp: kiểm tra sm version, nếu < 70 thì dùng CPU.
# Không cố reinstall PyTorch vì sẽ làm chậm kernel và không đảm bảo thành công.

print("\n==> Checking GPU compatibility...")
gpu_available = False
gpu_name = "N/A"

try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"    GPU detected: {gpu_info}")

        # Parse compute capability (e.g. "6.0" for P100, "7.5" for T4)
        parts = [p.strip() for p in gpu_info.split(",")]
        gpu_name = parts[0] if parts else "Unknown GPU"
        compute_cap_str = parts[2] if len(parts) >= 3 else "0.0"
        try:
            compute_cap = float(compute_cap_str)
        except ValueError:
            compute_cap = 0.0

        if compute_cap >= 7.0:
            # sm_70+ (T4, A100, V100, etc.) — tương thích với PyTorch 2.x
            print(f"    Compute capability {compute_cap} >= 7.0 — GPU compatible!")
            gpu_available = True
        else:
            # sm_60 (P100) — KHÔNG tương thích với PyTorch >= 2.0+cu12x
            print(f"    WARNING: Compute capability {compute_cap} < 7.0 (GPU: {gpu_name})")
            print(f"    PyTorch 2.x+cu121 requires sm_70+. Falling back to CPU.")
            print(f"    Training will be slower but will complete correctly.")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print("    WARNING: nvidia-smi failed — assuming no GPU")
except FileNotFoundError:
    print("    WARNING: nvidia-smi not found — assuming no GPU")
except subprocess.TimeoutExpired:
    print("    WARNING: nvidia-smi timed out — assuming no GPU")

# Double-check bằng torch nếu gpu_available=True
if gpu_available:
    try:
        import torch as _t
        if _t.cuda.is_available():
            # Thử một phép tính nhỏ để chắc chắn không crash
            _t.zeros(1).cuda()
            print(f"    PyTorch CUDA test OK: {_t.cuda.get_device_name(0)}, CUDA {_t.version.cuda}")
        else:
            print("    WARNING: torch.cuda.is_available() = False — falling back to CPU")
            gpu_available = False
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except Exception as e:
        print(f"    WARNING: CUDA test failed ({e}) — falling back to CPU")
        gpu_available = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

print(f"    Final device: {'GPU (' + gpu_name + ')' if gpu_available else 'CPU'}")

# Clone repo
PROJECT_DIR = "/kaggle/working/VietNerm"

if os.path.exists(PROJECT_DIR):
    print(f"\nPulling latest from {GITHUB_BRANCH}...")
    subprocess.run(["git", "pull", "origin", GITHUB_BRANCH], cwd=PROJECT_DIR, check=True)
else:
    print(f"\nCloning {GITHUB_REPO} (branch: {GITHUB_BRANCH})...")
    subprocess.run(
        ["git", "clone", "-b", GITHUB_BRANCH, GITHUB_REPO, PROJECT_DIR],
        check=True
    )

os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

# Install dependencies
print("\nInstalling project dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers", "datasets", "seqeval", "huggingface_hub",
     "pyyaml", "Jinja2", "numpy", "accelerate"],
    check=True
)
print("Dependencies installed.")

# ============================================================
# 3. DISCOVER DOC TYPES
# ============================================================

import yaml

with open("registry/documents.yaml") as f:
    registry = yaml.safe_load(f)

all_doc_types = list(registry["documents"].keys())

if DOC_TYPES_STR:
    doc_types = [d.strip() for d in DOC_TYPES_STR.split(",") if d.strip()]
    invalid = [d for d in doc_types if d not in all_doc_types]
    if invalid:
        print(f"ERROR: Invalid doc types: {invalid}")
        print(f"Available: {all_doc_types}")
        sys.exit(1)
else:
    doc_types = all_doc_types

print(f"\nDoc types to train ({len(doc_types)}):")
for dt in doc_types:
    info = registry["documents"][dt]
    print(f"  - {dt}: {info['name']}")

# ============================================================
# 4. PIPELINE: GENERATE → TRAIN → EVALUATE → PUBLISH
# ============================================================

results = {}
total_start = time.time()

for i, doc_type in enumerate(doc_types, 1):
    doc_name = registry["documents"][doc_type]["name"]
    print(f"\n{'=' * 60}")
    print(f"[{i}/{len(doc_types)}] {doc_type} — {doc_name}")
    print(f"{'=' * 60}")

    doc_start = time.time()
    doc_result = {"doc_type": doc_type, "name": doc_name, "status": "pending"}

    try:
        # ── Step 1: Generate data ──
        print(f"\n[1/4] Generating {DATASET_SIZE:,} samples...")
        ret = subprocess.run(
            [sys.executable, "synthetic/generate_dataset.py",
             "--doc", doc_type, "--size", str(DATASET_SIZE)],
            capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f"FAILED: Generate failed:\n{ret.stderr}")
            doc_result["status"] = "generate_failed"
            doc_result["error"] = ret.stderr[-500:]
            doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
            results[doc_type] = doc_result
            continue
        print("Generate done.")

        # ── Step 2: Train ──
        print(f"\n[2/4] Training ({TRAIN_EPOCHS} epochs, batch_size={BATCH_SIZE})...")
        # Truyền CUDA_VISIBLE_DEVICES vào subprocess để đảm bảo fallback CPU được kế thừa
        train_env = os.environ.copy()
        train_ret = subprocess.run(
            [sys.executable, "training/train.py",
             "--doc", doc_type,
             "--epochs", str(TRAIN_EPOCHS),
             "--batch_size", str(BATCH_SIZE)],
            env=train_env
        )
        if train_ret.returncode != 0:
            print("FAILED: Training failed!")
            doc_result["status"] = "train_failed"
            doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
            results[doc_type] = doc_result
            continue
        print("Training done.")

        # ── Step 3: Evaluate — đọc F1 từ trainer_state.json ──
        print(f"\n[3/4] Evaluating (min F1={MIN_F1})...")
        model_dir = Path(f"models/phobert/{doc_type}")
        best_f1 = None  # None = chưa đọc được, khác với 0.0 = đọc được nhưng thấp
        trainer_state_path = model_dir / "trainer_state.json"

        if trainer_state_path.exists():
            try:
                with open(trainer_state_path) as f:
                    state = json.load(f)
                for entry in state.get("log_history", []):
                    # HuggingFace Trainer log key là "eval_f1" (metric_for_best_model="f1")
                    f1_val = entry.get("eval_f1")
                    if f1_val is not None:
                        if best_f1 is None or f1_val > best_f1:
                            best_f1 = f1_val
                if best_f1 is not None:
                    print(f"    Best F1 (from trainer_state.json): {best_f1:.4f}")
                else:
                    print("    WARNING: trainer_state.json exists but no eval_f1 entries found")
                    print(f"    Log history keys: {[list(e.keys()) for e in state.get('log_history', [])[:3]]}")
            except Exception as e:
                print(f"    WARNING: Could not read trainer_state.json: {e}")
        else:
            print(f"    WARNING: trainer_state.json not found at {trainer_state_path}")

        # Nếu không đọc được F1 từ trainer_state, thử chạy evaluate riêng
        if best_f1 is None:
            print("    Attempting standalone evaluation...")
            eval_ret = subprocess.run(
                [sys.executable, "-c", f"""
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from training.trainer import PhoBERTNERTrainer
import yaml

config_path = Path('training/config/{doc_type}.yaml')
if not config_path.exists():
    config_path = Path('training/config/default.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

trainer_obj = PhoBERTNERTrainer(config)
dataset_dir = Path('datasets/ner/{doc_type}')
model_dir = Path('models/phobert/{doc_type}')

from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset, Features, Sequence, Value
from training.trainer import _tokenize_and_align_labels, _compute_metrics
import numpy as np

_, test_samples, label_list = trainer_obj.load_dataset(dataset_dir)
label2id = {{l: i for i, l in enumerate(label_list)}}
id2label = {{i: l for i, l in enumerate(label_list)}}

tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

features = Features({{"tokens": Sequence(Value("string")), "ner_tags": Sequence(Value("string"))}})
test_hf = Dataset.from_list([{{"tokens": s["tokens"], "ner_tags": s["ner_tags"]}} for s in test_samples[:200]], features=features)
test_tok = test_hf.map(_tokenize_and_align_labels, fn_kwargs={{"tokenizer": tokenizer, "label2id": label2id, "max_length": 256}}, batched=True)

args = TrainingArguments('/tmp/eval_only', per_device_eval_batch_size=16, no_cuda=True)
t = Trainer(model=model, args=args, eval_dataset=test_tok, compute_metrics=lambda p: _compute_metrics(p, id2label))
metrics = t.evaluate()
print(json.dumps({{"eval_f1": metrics.get("eval_f1", 0.0)}}))
"""],
                capture_output=True, text=True, env=train_env
            )
            if eval_ret.returncode == 0:
                try:
                    last_line = [l for l in eval_ret.stdout.strip().split("\n") if l.strip().startswith("{")]
                    if last_line:
                        eval_metrics = json.loads(last_line[-1])
                        best_f1 = eval_metrics.get("eval_f1", 0.0)
                        print(f"    F1 from standalone eval: {best_f1:.4f}")
                except Exception as e2:
                    print(f"    WARNING: Could not parse standalone eval output: {e2}")
            else:
                print(f"    WARNING: Standalone eval failed: {eval_ret.stderr[-200:]}")

        # Fallback: nếu vẫn không có F1, đặt = 0.0 và ghi rõ lý do
        if best_f1 is None:
            best_f1 = 0.0
            print("    WARNING: Could not determine F1 — defaulting to 0.0")

        doc_result["f1"] = best_f1

        if best_f1 < MIN_F1:
            print(f"    WARNING: F1 ({best_f1:.4f}) < threshold ({MIN_F1}) — skip publish")
            doc_result["status"] = "low_f1"
            doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
            results[doc_type] = doc_result
            continue
        print("    Evaluation passed.")

        # ── Step 4: Publish to HuggingFace ──
        if not HF_TOKEN:
            print("    WARNING: HF_TOKEN not set — skip publish")
            doc_result["status"] = "no_hf_token"
            doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
            results[doc_type] = doc_result
            continue

        print(f"\n[4/4] Publishing to HuggingFace Hub...")
        model_repo = f"{HF_USERNAME}/phobert-{doc_type}-ner"
        dataset_repo = f"{HF_USERNAME}/vietnerm-{doc_type}-dataset"
        private_flag = ["--private"] if HF_PRIVATE else []

        # Push model
        print(f"  Model  -> {model_repo}")
        ret = subprocess.run(
            [sys.executable, "huggingface/push_model.py",
             "--doc", doc_type, "--repo", model_repo,
             "--token", HF_TOKEN] + private_flag,
            capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f"FAILED: Push model failed:\n{ret.stderr[-300:]}")
            doc_result["status"] = "publish_failed"
            doc_result["error"] = ret.stderr[-500:]
            doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
            results[doc_type] = doc_result
            continue

        # Push dataset
        print(f"  Dataset -> {dataset_repo}")
        ret = subprocess.run(
            [sys.executable, "huggingface/push_dataset.py",
             "--doc", doc_type, "--repo", dataset_repo,
             "--token", HF_TOKEN] + private_flag,
            capture_output=True, text=True
        )
        if ret.returncode != 0:
            print(f"WARNING: Push dataset failed (model already published):\n{ret.stderr[-200:]}")

        doc_result["status"] = "published"
        doc_result["model_url"] = f"https://huggingface.co/{model_repo}"
        doc_result["dataset_url"] = f"https://huggingface.co/datasets/{dataset_repo}"
        print(f"Published!")
        print(f"  Model:   https://huggingface.co/{model_repo}")
        print(f"  Dataset: https://huggingface.co/datasets/{dataset_repo}")

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        doc_result["status"] = "error"
        doc_result["error"] = str(e)

    doc_result["time_minutes"] = round((time.time() - doc_start) / 60, 1)
    results[doc_type] = doc_result

# ============================================================
# 5. SAVE RESULTS & SUMMARY
# ============================================================

total_minutes = round((time.time() - total_start) / 60, 1)

# Save results JSON (sẽ được download từ Kaggle output)
output_path = "/kaggle/working/train_results.json"
with open(output_path, "w") as f:
    json.dump({
        "timestamp": datetime.now().isoformat(),
        "total_minutes": total_minutes,
        "gpu_used": gpu_available,
        "gpu_name": gpu_name,
        "config": {
            "dataset_size": DATASET_SIZE,
            "train_epochs": TRAIN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "min_f1": MIN_F1,
            "hf_username": HF_USERNAME,
            "github_branch": GITHUB_BRANCH,
        },
        "results": results,
    }, f, indent=2, ensure_ascii=False)

# Copy models to /kaggle/working/ for download
import shutil
for doc_type, r in results.items():
    if r["status"] in ("published", "low_f1", "no_hf_token") or r.get("f1", 0) > 0:
        model_src = Path(f"models/phobert/{doc_type}")
        model_dst = Path(f"/kaggle/working/models/phobert/{doc_type}")
        if model_src.exists():
            shutil.copytree(model_src, model_dst, dirs_exist_ok=True)
            print(f"Copied model: {doc_type} -> {model_dst}")

# Summary
print("\n" + "=" * 60)
print("PIPELINE RESULTS")
print("=" * 60)
print(f"Total time: {total_minutes} minutes")
print(f"Dataset size: {DATASET_SIZE:,}")
print(f"Train epochs: {TRAIN_EPOCHS}")
print(f"Device: {'GPU (' + gpu_name + ')' if gpu_available else 'CPU (GPU incompatible or unavailable)'}")
print()

print(f"{'Doc Type':<25} {'Status':<18} {'F1':>8} {'Time':>8}")
print("-" * 65)

for doc_type, r in results.items():
    f1_val = r.get("f1")
    f1_str = f"{f1_val:.4f}" if f1_val is not None and f1_val > 0 else "—"
    time_str = f"{r.get('time_minutes', 0)}m"
    print(f"  {doc_type:<23} {r['status']:<18} {f1_str:>8} {time_str:>8}")

published = [r for r in results.values() if r["status"] == "published"]
print(f"\n{len(published)}/{len(results)} models published successfully!")
print(f"Results saved to: {output_path}")
