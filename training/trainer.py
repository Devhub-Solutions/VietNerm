"""PhoBERT NER Trainer class.

Refactored from train/train_giaravien.py — preserves all training logic
including tokenization alignment, label validation, metrics computation,
and early stopping.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from datasets import Dataset, Features, Sequence, Value
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class PhoBERTNERTrainer:
    """Universal PhoBERT NER trainer for any document type.

    Loads datasets from the standard NER format (train.json, test.json, labels.json),
    trains a PhoBERT model with BIO tagging, and saves the model.
    """

    # Config keys that must be numeric (YAML may parse '2e-5' as str)
    _FLOAT_KEYS = {"learning_rate", "weight_decay", "train_split", "valid_split"}
    _INT_KEYS = {
        "max_length", "num_train_epochs", "per_device_train_batch_size",
        "per_device_eval_batch_size", "warmup_steps", "logging_steps",
        "early_stopping_patience", "save_total_limit",
    }
    _INFERENCE_KEEP_FILES = {
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",  # fallback for older Transformers
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "bpe.codes",
        "added_tokens.json",
        "label_map.json",
        "README.md",
        ".gitattributes",
    }

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the trainer with configuration.

        Args:
            config: Training configuration dictionary. Expected keys:
                model_name, max_length, num_train_epochs, learning_rate,
                per_device_train_batch_size, weight_decay, warmup_steps, etc.
        """
        self._config = self._coerce_config_types(config)
        self._model_name: str = self._config.get("model_name", "vinai/phobert-base")
        self._max_length: int = self._config.get("max_length", 512)
        self._tokenizer = None
        self._model = None
        self._label_list: List[str] = []
        self._label2id: Dict[str, int] = {}
        self._id2label: Dict[int, str] = {}

    @classmethod
    def _coerce_config_types(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure numeric config values have correct types.

        YAML safe_load parses '2e-5' as a string (not float).
        This method coerces known numeric keys to their expected types.
        """
        out = dict(config)
        for key in cls._FLOAT_KEYS:
            if key in out and not isinstance(out[key], float):
                try:
                    out[key] = float(out[key])
                except (ValueError, TypeError):
                    pass
        for key in cls._INT_KEYS:
            if key in out and not isinstance(out[key], int):
                try:
                    out[key] = int(out[key])
                except (ValueError, TypeError):
                    pass
        return out

    @classmethod
    def from_config_file(cls, config_path: Path) -> "PhoBERTNERTrainer":
        """Create a trainer from a YAML config file.

        Args:
            config_path: Path to YAML configuration file.

        Returns:
            Configured PhoBERTNERTrainer instance.
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def load_dataset(
        self,
        dataset_dir: Path,
    ) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Load NER dataset from directory.

        Args:
            dataset_dir: Directory containing train.json, test.json, labels.json.

        Returns:
            Tuple of (train_samples, test_samples, label_list).
        """
        dataset_dir = Path(dataset_dir)

        with open(dataset_dir / "train.json", "r", encoding="utf-8") as f:
            train_samples = json.load(f)

        with open(dataset_dir / "test.json", "r", encoding="utf-8") as f:
            test_samples = json.load(f)

        with open(dataset_dir / "labels.json", "r", encoding="utf-8") as f:
            labels_data = json.load(f)

        label_list = labels_data["labels"]
        return train_samples, test_samples, label_list

    def train(
        self,
        dataset_dir: Path,
        output_dir: Path,
        real_data_path: Optional[Path] = None,
        real_data_weight: int = 3,
    ) -> Tuple[Any, Any, Dict[int, str]]:
        """Run the full training pipeline.

        Args:
            dataset_dir: Path to datasets/ner/{doc}/ directory.
            output_dir: Path to save trained model.
            real_data_path: Optional path to real annotation data.
            real_data_weight: How many times to duplicate real data.

        Returns:
            Tuple of (tokenizer, model, id2label).
        """
        print("==> Loading dataset...")
        train_samples, test_samples, label_list = self.load_dataset(dataset_dir)

        # Optionally merge real data
        if real_data_path and real_data_path.exists():
            real_samples = self._load_real_data(real_data_path, label_list)
            if real_samples:
                train_samples = train_samples + real_samples * real_data_weight
                print(f"    Real data (x{real_data_weight}): "
                      f"{len(real_samples) * real_data_weight} samples")

        # Split train into train/valid
        random.shuffle(train_samples)
        train_split = self._config.get("train_split", 0.8)
        valid_split = self._config.get("valid_split", 0.1)

        # Use the test set from file, split train into train/valid
        n = len(train_samples)
        valid_size = int(n * valid_split / (train_split + valid_split))
        valid_samples = train_samples[:valid_size]
        actual_train = train_samples[valid_size:]

        print(f"    Train: {len(actual_train)}, Valid: {len(valid_samples)}, "
              f"Test: {len(test_samples)}")

        # Set up labels
        self._label_list = label_list
        self._label2id = {l: i for i, l in enumerate(label_list)}
        self._id2label = {i: l for i, l in enumerate(label_list)}
        print(f"    Labels ({len(label_list)}): {label_list}")

        # Load tokenizer and model
        print(f"==> Loading tokenizer & {self._model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, use_fast=False
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            self._model_name,
            num_labels=len(label_list),
            id2label=self._id2label,
            label2id=self._label2id,
        )

        # Build HF datasets with sliding window tokenization
        print("==> Tokenizing datasets...")

        stride = self._config.get("stride", 128)
        print(f"    Sliding window: max_length={self._max_length}, stride={stride}")

        # NOTE: We do NOT use Dataset.map() for sliding window because map(batched=True)
        # requires output row count == input row count. Sliding window produces MORE rows
        # than input (1 long doc → N windows), causing PyArrow ArrowInvalid error.
        # Solution: build the windowed dataset directly with Dataset.from_list().
        train_windows = _build_windowed_dataset(
            actual_train, self._tokenizer, self._label2id,
            self._max_length, stride,
        )
        valid_windows = _build_windowed_dataset(
            valid_samples, self._tokenizer, self._label2id,
            self._max_length, stride,
        )

        tok_features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int8")),
            "labels": Sequence(Value("int32")),
        })
        train_tok = Dataset.from_list(train_windows, features=tok_features)
        valid_tok = Dataset.from_list(valid_windows, features=tok_features)
        print(f"    Windows: train={len(train_tok)}, valid={len(valid_tok)}")

        # Validate label IDs before training
        print("==> Validating label IDs...")
        _validate_label_ids(train_tok, len(label_list), "train")
        _validate_label_ids(valid_tok, len(label_list), "valid")

        # Training arguments
        print("==> Training...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Xác định device: nếu CUDA_VISIBLE_DEVICES="" thì force CPU
        # (được set bởi kernel khi GPU không tương thích, ví dụ P100 sm_60)
        _cuda_disabled = os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""
        _use_gpu = torch.cuda.is_available() and not _cuda_disabled
        print(f"    Trainer device: {'GPU (fp16)' if _use_gpu else 'CPU (no_cuda=True)'}")

        # Disable wandb/tensorboard — no API keys available in CI/CD
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"

        args = TrainingArguments(
            str(output_dir),
            report_to="none",
            learning_rate=self._config.get("learning_rate", 2e-5),
            per_device_train_batch_size=self._config.get(
                "per_device_train_batch_size", 16
            ),
            per_device_eval_batch_size=self._config.get(
                "per_device_eval_batch_size", 16
            ),
            num_train_epochs=self._config.get("num_train_epochs", 7),
            weight_decay=self._config.get("weight_decay", 0.01),
            eval_strategy=self._config.get("eval_strategy", "epoch"),
            save_strategy=self._config.get("save_strategy", "epoch"),
            load_best_model_at_end=self._config.get(
                "load_best_model_at_end", True
            ),
            metric_for_best_model=self._config.get(
                "metric_for_best_model", "f1"
            ),
            logging_steps=self._config.get("logging_steps", 50),
            fp16=_use_gpu,
            use_cpu=_cuda_disabled,
            optim=self._config.get("optim", "adamw_torch"),
            warmup_steps=self._config.get("warmup_steps", 100),
            save_total_limit=self._config.get("save_total_limit", 1),
            save_only_model=self._config.get("save_only_model", True),
        )

        id2label = self._id2label

        callbacks = []
        patience = self._config.get("early_stopping_patience")
        if patience:
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=patience
            ))

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=valid_tok,
            processing_class=self._tokenizer,
            compute_metrics=lambda p: _compute_metrics(p, id2label),
            callbacks=callbacks,
        )
        trainer.train()

        # Save model, tokenizer, trainer state, and label map
        print(f"==> Saving model and state to {output_dir}...")
        trainer.save_model(str(output_dir))
        trainer.save_state()
        self._tokenizer.save_pretrained(str(output_dir))

        label_map_path = output_dir / "label_map.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(
                {"id2label": self._id2label, "label2id": self._label2id},
                f, ensure_ascii=False, indent=2,
            )

        clean_dir = output_dir / "inference"
        self._export_inference_dir(output_dir, clean_dir)
        print(f"==> Exported clean inference artifacts to {clean_dir}")

        print("==> Done!")
        return self._tokenizer, self._model, self._id2label

    def _export_inference_dir(self, source_dir: Path, target_dir: Path) -> None:
        """Export a clean inference-only directory from training output."""
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        for file_name in sorted(self._INFERENCE_KEEP_FILES):
            src = source_dir / file_name
            if src.exists() and src.is_file():
                shutil.copy2(src, target_dir / file_name)

    def _load_real_data(
        self,
        json_path: Path,
        label_list: List[str],
    ) -> List[Dict]:
        """Load real annotation data and convert to BIO format."""
        from synthetic.generate_dataset import tokenize_with_offsets, assign_bio_labels

        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        entity_types = set(label_list) - {"O"}
        valid_types = set()
        for label in entity_types:
            # Extract entity type from B-xxx or I-xxx
            if label.startswith("B-") or label.startswith("I-"):
                valid_types.add(label[2:])

        samples = []
        for ann in annotations:
            text = ann.get("text", "")
            entities = ann.get("entities", [])
            if not text:
                continue

            tokens_with_pos = tokenize_with_offsets(text)
            if not tokens_with_pos:
                continue

            # Filter entities to only known types
            valid_entities = []
            for ent in entities:
                label = ent.get("label", "")
                if label in valid_types:
                    valid_entities.append(ent)

            tokens = [t for t, _, _ in tokens_with_pos]
            tags = assign_bio_labels(tokens_with_pos, valid_entities)
            samples.append({"tokens": tokens, "ner_tags": tags})

        print(f"    Loaded {len(samples)} real samples from {json_path}")
        return samples


# ═══════════════════════════════════════════
# Module-level functions (used by Trainer callbacks)
# ═══════════════════════════════════════════

# PhoBERT hard limit: max_position_embeddings = 258 (256 + CLS + SEP)
PHOBERT_MAX_TOKENS = 256  # usable tokens excluding CLS/SEP


def _encode_sample_to_subwords(
    tokens: List[str],
    ner_tags: List[str],
    tokenizer: Any,
    label2id: Dict[str, int],
    id_to_label_keys: List[str],
    num_labels: int,
) -> Tuple[List[int], List[int]]:
    """Encode a word-level sample to subword ids and aligned label ids.

    Returns:
        (subword_ids, subword_labels) — flat lists without CLS/SEP.
        Labels for non-first subwords are -100 (ignored in loss).
    """
    subword_ids: List[int] = []
    subword_labels: List[int] = []

    for text, tag in zip(tokens, ner_tags):
        # Guard: if tag is int (HF auto-encode), convert back to string
        if isinstance(tag, int):
            tag = (
                id_to_label_keys[tag]
                if tag < len(id_to_label_keys)
                else "O"
            )

        word_tokens = tokenizer.encode(text, add_special_tokens=False)
        if not word_tokens:
            continue

        label_id = label2id.get(tag, label2id["O"])
        if not (0 <= label_id < num_labels):
            label_id = label2id["O"]

        subword_ids.extend(word_tokens)
        subword_labels.append(label_id)
        subword_labels.extend([-100] * (len(word_tokens) - 1))

    return subword_ids, subword_labels


def _make_windows(
    subword_ids: List[int],
    subword_labels: List[int],
    tokenizer: Any,
    max_length: int,
    stride: int,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """Slice a long subword sequence into overlapping windows.

    Each window has at most `max_length` tokens (including CLS/SEP).
    Windows overlap by `stride` subword tokens so entities near boundaries
    are seen in multiple windows during training.

    Returns:
        List of (input_ids, attention_mask, labels) tuples, each padded
        to exactly `max_length`.
    """
    usable = max_length - 2  # reserve slots for CLS and SEP
    windows = []
    step = max(1, usable - stride)
    total = len(subword_ids)

    start = 0
    while start < total:
        end = min(start + usable, total)
        chunk_ids = subword_ids[start:end]
        chunk_labels = subword_labels[start:end]

        input_ids = [tokenizer.cls_token_id] + chunk_ids + [tokenizer.sep_token_id]
        labels = [-100] + chunk_labels + [-100]

        padding_len = max_length - len(input_ids)
        mask = [1] * len(input_ids) + [0] * padding_len
        input_ids += [tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len

        windows.append((input_ids, mask, labels))

        if end == total:
            break
        start += step

    return windows


def _build_windowed_dataset(
    samples: List[Dict],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int,
    stride: int,
) -> List[Dict[str, List[int]]]:
    """Convert word-level samples to a flat list of windowed token dicts.

    Unlike Dataset.map(), this function can freely produce more output rows
    than input rows (N windows per long document). The result is a plain
    Python list suitable for Dataset.from_list().

    Args:
        samples: List of {'tokens': [...], 'ner_tags': [...]} dicts.
        tokenizer: PhoBERT tokenizer.
        label2id: Label string → int mapping.
        max_length: Max tokens per window (must be ≤ 258 for PhoBERT).
        stride: Overlap in subword tokens between consecutive windows.

    Returns:
        List of {'input_ids': [...], 'attention_mask': [...], 'labels': [...]}
        dicts, one per window.
    """
    num_labels = len(label2id)
    id_to_label_keys = list(label2id.keys())
    result: List[Dict[str, List[int]]] = []

    for sample in samples:
        tokens = sample["tokens"]
        ner_tags = sample.get("ner_tags", sample.get("labels", []))

        subword_ids, subword_labels = _encode_sample_to_subwords(
            tokens, ner_tags, tokenizer, label2id, id_to_label_keys, num_labels
        )
        if not subword_ids:
            continue

        windows = _make_windows(
            subword_ids, subword_labels, tokenizer, max_length, stride
        )
        for input_ids, mask, labels in windows:
            result.append({
                "input_ids": input_ids,
                "attention_mask": mask,
                "labels": labels,
            })

    return result


def _tokenize_and_align_labels(
    examples: Dict,
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int = 256,
    stride: int = 128,
) -> Dict[str, List]:
    """Tokenize with PhoBERT and align BIO labels to subword tokens.

    Supports sliding window for documents longer than PhoBERT's 256-token
    hard limit. Each long document is split into overlapping windows of
    `max_length` tokens with `stride` overlap. This multiplies the number
    of training samples but ensures no content is silently dropped.

    Args:
        examples: Batched HF dataset dict with 'tokens' and 'ner_tags'.
        tokenizer: PhoBERT tokenizer.
        label2id: Label string → int mapping.
        max_length: Max tokens per window (must be <= 258 for PhoBERT).
        stride: Overlap in subword tokens between consecutive windows.
    """
    num_labels = len(label2id)
    id_to_label_keys = list(label2id.keys())

    all_input_ids, all_attention_mask, all_labels = [], [], []

    for i, tokens in enumerate(examples["tokens"]):
        ner_tags = examples["ner_tags"][i]

        # Step 1: encode entire sample to flat subword sequence
        subword_ids, subword_labels = _encode_sample_to_subwords(
            tokens, ner_tags, tokenizer, label2id, id_to_label_keys, num_labels
        )

        if not subword_ids:
            continue

        # Step 2: slice into windows (handles both short and long documents)
        windows = _make_windows(
            subword_ids, subword_labels, tokenizer, max_length, stride
        )

        for input_ids, mask, labels in windows:
            # Sanity check
            bad = [l for l in labels if l != -100 and not (0 <= l < num_labels)]
            assert not bad, (
                f"Sample {i}: label out of range! bad={bad}, num_labels={num_labels}"
            )
            all_input_ids.append(input_ids)
            all_attention_mask.append(mask)
            all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }


def _validate_label_ids(dataset: Dataset, num_labels: int, name: str) -> None:
    """Validate all label IDs are within bounds before GPU training."""
    bad_count = 0
    for row in dataset:
        for l in row["labels"]:
            if l != -100 and not (0 <= l < num_labels):
                bad_count += 1
    if bad_count:
        raise ValueError(
            f"[{name}] {bad_count} label IDs outside [0, {num_labels - 1}]!"
        )
    print(f"    [{name}] OK — all label IDs valid (num_labels={num_labels})")


def _compute_metrics(p: Any, id2label: Dict[int, str]) -> Dict[str, float]:
    """Compute seqeval F1 metrics for NER evaluation.

    Handles cases where p.predictions is a tuple (e.g. including hidden states).
    """
    preds, labels = p
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=-1)
    true_labels: List[List[str]] = []
    true_preds: List[List[str]] = []

    for pred_seq, lab_seq in zip(preds, labels):
        cur_l: List[str] = []
        cur_p: List[str] = []
        for p_id, l_id in zip(pred_seq, lab_seq):
            if l_id == -100:
                continue
            cur_l.append(id2label[l_id])
            cur_p.append(id2label[p_id])
        true_labels.append(cur_l)
        true_preds.append(cur_p)

    overall_f1 = f1_score(true_labels, true_preds)

    try:
        report = classification_report(
            true_labels, true_preds, output_dict=True
        )
        per_entity: Dict[str, float] = {}
        # seqeval report keys are entity types (e.g. 'full_name'), not starting with B-
        # We exclude 'micro avg', 'macro avg', etc.
        skip_keys = {"micro avg", "macro avg", "weighted avg"}
        for k, v in report.items():
            if isinstance(v, dict) and k not in skip_keys:
                per_entity[k] = v.get("f1-score", 0.0)

        if per_entity:
            print("\n[Per-entity F1]")
            for etype, score in sorted(per_entity.items()):
                print(f"  {etype:35s}: {score:.4f}")
    except Exception as e:
        print(f"    (Metric display error: {e})")

    return {"f1": overall_f1}
