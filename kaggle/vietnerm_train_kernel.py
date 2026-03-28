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
    # Thử Kaggle Secrets trước
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

# Kiểm tra GPU
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"GPU: {result.stdout.strip()}")
    else:
        print("WARNING: GPU not detected!")
except FileNotFoundError:
    print("WARNING: nvidia-smi not found!")

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

# Install dependencies — handle GPU compatibility
print("\nChecking GPU CUDA compatibility...")
gpu_compatible = False
try:
    import torch as _torch_check
    if _torch_check.cuda.is_available():
        _torch_check.zeros(1).cuda()
        gpu_compatible = True
        print(f"GPU OK: {_torch_check.cuda.get_device_name(0)}, CUDA {_torch_check.version.cuda}")
except Exception as e:
    print(f"GPU not compatible with pre-installed PyTorch: {e}")

if not gpu_compatible:
    # P100 (sm_60) cần PyTorch build với CUDA 11.8
    # Thử cài PyTorch cu118 — version cuối hỗ trợ sm_60
    print("Reinstalling PyTorch with CUDA 11.8 (supports P100/sm_60)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "torch==2.2.2+cu118", "torchvision==0.17.2+cu118",
         "--index-url", "https://download.pytorch.org/whl/cu118"],
        check=False  # Nếu fail, sẽ fallback CPU
    )
    # Verify lại
    try:
        import importlib
        import torch
        importlib.reload(torch)
        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            print(f"GPU OK after reinstall: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: GPU still not available, falling back to CPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except Exception as e2:
        print(f"WARNING: GPU fallback failed ({e2}), using CPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("\nInstalling project dependencies...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "-r", "requirements.txt", "accelerate", "--no-deps"],
    check=False
)
# Cài lại những deps còn thiếu (nhưng không override torch)
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
            results[doc_type] = doc_result
            continue
        print("Generate done.")

        # ── Step 2: Train ──
        print(f"\n[2/4] Training ({TRAIN_EPOCHS} epochs, batch_size={BATCH_SIZE})...")
        train_ret = subprocess.run(
            [sys.executable, "training/train.py",
             "--doc", doc_type,
             "--epochs", str(TRAIN_EPOCHS),
             "--batch_size", str(BATCH_SIZE)]
        )
        if train_ret.returncode != 0:
            print("FAILED: Training failed!")
            doc_result["status"] = "train_failed"
            results[doc_type] = doc_result
            continue
        print("Training done.")

        # ── Step 3: Evaluate ──
        print(f"\n[3/4] Evaluating (min F1={MIN_F1})...")
        model_dir = f"models/phobert/{doc_type}"
        best_f1 = 0.0
        trainer_state_path = Path(model_dir) / "trainer_state.json"
        if trainer_state_path.exists():
            with open(trainer_state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                f1 = entry.get("eval_f1", 0.0)
                if f1 > best_f1:
                    best_f1 = f1

        doc_result["f1"] = best_f1
        print(f"  Best F1: {best_f1:.4f}")

        if best_f1 < MIN_F1:
            print(f"WARNING: F1 ({best_f1:.4f}) < threshold ({MIN_F1}) — skip publish")
            doc_result["status"] = "low_f1"
            results[doc_type] = doc_result
            continue
        print("Evaluation passed.")

        # ── Step 4: Publish to HuggingFace ──
        if not HF_TOKEN:
            print("WARNING: HF_TOKEN not set — skip publish")
            doc_result["status"] = "no_hf_token"
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
    if r["status"] == "published" or r.get("f1", 0) > 0:
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
print()

print(f"{'Doc Type':<25} {'Status':<18} {'F1':>8} {'Time':>8}")
print("-" * 65)

for doc_type, r in results.items():
    f1_str = f"{r.get('f1', 0):.4f}" if r.get("f1") else "—"
    time_str = f"{r.get('time_minutes', 0)}m"
    print(f"  {doc_type:<23} {r['status']:<18} {f1_str:>8} {time_str:>8}")

published = [r for r in results.values() if r["status"] == "published"]
print(f"\n{len(published)}/{len(results)} models published successfully!")
print(f"Results saved to: {output_path}")