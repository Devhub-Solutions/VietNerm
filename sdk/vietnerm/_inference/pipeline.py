"""
NER Pipeline - Load PhoBERT model and run NER prediction.

Supports loading from:
  - Local path: /path/to/model/
  - HuggingFace Hub: username/phobert-{doc}-ner

Sliding Window:
  PhoBERT has a hard limit of 258 position embeddings (256 + CLS + SEP).
  For long documents, the pipeline automatically splits tokens into overlapping
  windows (stride=128 by default), runs inference on each window, then merges
  predictions at overlapping positions using average confidence scoring.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .postprocess import merge_subtoken_predictions, clean_entity_boundaries, compute_confidence

# PhoBERT hard limit: 256 content tokens + CLS + SEP = 258
_PHOBERT_MAX_CONTENT = 254  # leave 2 slots for CLS/SEP


def _is_hub_id(model_path: str) -> bool:
    """Return True if model_path looks like a HuggingFace Hub repo ID (user/repo)."""
    p = Path(model_path)
    if p.exists():
        return False
    parts = model_path.strip("/").split("/")
    return len(parts) == 2


def _load_label_map_from_hub(repo_id: str) -> Optional[Dict]:
    """Try to download label_map.json from a HuggingFace Hub repo.

    Returns the parsed dict, or None if the file does not exist.
    """
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_id, filename="label_map.json")
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


class NERPipeline:
    """PhoBERT-based NER pipeline for Vietnamese document entity extraction.

    Automatically applies sliding window for documents longer than PhoBERT's
    256-token limit. Predictions at overlapping positions are merged using
    average confidence scoring.

    Args:
        model_path: Local path or HuggingFace hub ID for the model.
        device: Device to run inference on ('cpu', 'cuda', or 'auto').
        max_length: Maximum token sequence length (default 256, PhoBERT limit).
        stride: Sliding window stride in subword tokens (default 128 = 50% overlap).
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 256,
        stride: int = 128,
    ) -> None:
        self.model_path = model_path
        self.max_length = min(max_length, 256)  # enforce PhoBERT hard limit
        self.stride = stride

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.label2id: Dict[str, int] = {
            v: int(k) for k, v in self.model.config.id2label.items()
        }

        # Try loading label_map.json — local path or HuggingFace Hub
        label_map: Optional[Dict] = None
        if _is_hub_id(model_path):
            label_map = _load_label_map_from_hub(model_path)
        else:
            label_map_path = Path(model_path) / "label_map.json"
            if label_map_path.exists():
                with open(label_map_path, "r", encoding="utf-8") as f:
                    label_map = json.load(f)

        if label_map:
            if "id2label" in label_map:
                self.id2label = {int(k): v for k, v in label_map["id2label"].items()}
            if "label2id" in label_map:
                self.label2id = {k: int(v) for k, v in label_map["label2id"].items()}

    @staticmethod
    def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
        """Whitespace tokenization with character offset tracking.

        Args:
            text: Input text to tokenize.

        Returns:
            List of (token, start_offset, end_offset) tuples.
        """
        flat = text.replace("\n", " ")
        result: List[Tuple[str, int, int]] = []
        pos = 0
        for tok in flat.split(" "):
            if not tok:
                pos += 1
                continue
            start = flat.find(tok, pos)
            if start == -1:
                pos += len(tok) + 1
                continue
            end = start + len(tok)
            result.append((tok, start, end))
            pos = end + 1
        return result

    def _encode_tokens(
        self, tokens: List[str]
    ) -> Tuple[List[int], List[Optional[Tuple[int, int]]]]:
        """Encode a list of word tokens into subword IDs.

        Returns:
            - subword_ids: flat list of subword token IDs (WITHOUT CLS/SEP)
            - token_sw_ranges: for each word token, (sw_start, sw_end) index
              into subword_ids, or None if the token produced no subwords.
        """
        subword_ids: List[int] = []
        token_sw_ranges: List[Optional[Tuple[int, int]]] = []

        for tok in tokens:
            word_ids = self.tokenizer.encode(tok, add_special_tokens=False)
            if not word_ids:
                token_sw_ranges.append(None)
            else:
                sw_start = len(subword_ids)
                subword_ids.extend(word_ids)
                token_sw_ranges.append((sw_start, sw_start + len(word_ids)))

        return subword_ids, token_sw_ranges

    def _run_window(self, window_ids: List[int]) -> Tuple[List[str], List[float]]:
        """Run a single forward pass on a window of subword IDs.

        Args:
            window_ids: Subword token IDs (WITHOUT CLS/SEP). Will be wrapped
                with CLS/SEP and padded to max_length.

        Returns:
            - labels: predicted BIO label for each position in window_ids
            - confidences: confidence score for each position
        """
        # Wrap with CLS/SEP
        full_ids = [self.tokenizer.cls_token_id] + window_ids + [self.tokenizer.sep_token_id]
        n = len(full_ids)

        # Pad to max_length
        pad_id = self.tokenizer.pad_token_id or 0
        attention_mask = [1] * n + [0] * (self.max_length - n)
        full_ids = full_ids + [pad_id] * (self.max_length - n)

        input_tensor = torch.tensor([full_ids], dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor)

        logits = outputs.logits[0]  # (max_length, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).tolist()
        max_probs = probs.max(-1).values.tolist()

        # Positions 1..len(window_ids) correspond to actual content (skip CLS at 0)
        labels = []
        confidences = []
        for i in range(len(window_ids)):
            pos = i + 1  # +1 for CLS
            labels.append(self.id2label.get(pred_ids[pos], "O"))
            confidences.append(float(max_probs[pos]))

        return labels, confidences

    def _predict_with_sliding_window(
        self,
        subword_ids: List[int],
        token_sw_ranges: List[Optional[Tuple[int, int]]],
    ) -> Tuple[List[str], List[float]]:
        """Run sliding window inference over subword IDs.

        Splits subword_ids into overlapping windows of size max_length-2
        (leaving room for CLS/SEP), runs forward pass on each, then merges
        predictions at overlapping positions using average confidence.

        At overlapping positions, the label with the HIGHEST average confidence
        across all windows is chosen.

        Args:
            subword_ids: All subword token IDs for the document.
            token_sw_ranges: Mapping from word token index to subword range.

        Returns:
            - token_labels: BIO label for each word token
            - token_confidences: confidence for each word token
        """
        content_size = self.max_length - 2  # slots available per window (excl CLS/SEP)
        stride = self.stride
        total_sw = len(subword_ids)

        # Collect per-subword-position predictions: list of (label, confidence)
        sw_predictions: Dict[int, List[Tuple[str, float]]] = defaultdict(list)

        # Slide windows
        start = 0
        while start < total_sw:
            end = min(start + content_size, total_sw)
            window_ids = subword_ids[start:end]

            labels, confidences = self._run_window(window_ids)

            for local_i, (lbl, conf) in enumerate(zip(labels, confidences)):
                global_i = start + local_i
                sw_predictions[global_i].append((lbl, conf))

            if end >= total_sw:
                break
            start += stride

        # Merge: for each subword position, pick label with highest avg confidence
        merged_labels: List[str] = []
        merged_confs: List[float] = []
        for sw_i in range(total_sw):
            preds = sw_predictions.get(sw_i, [("O", 0.0)])
            # Group by label, average confidence per label
            label_conf: Dict[str, List[float]] = defaultdict(list)
            for lbl, conf in preds:
                label_conf[lbl].append(conf)
            best_label = max(label_conf, key=lambda l: np.mean(label_conf[l]))
            best_conf = float(np.mean(label_conf[best_label]))
            merged_labels.append(best_label)
            merged_confs.append(best_conf)

        # Map subword predictions back to word token level
        token_labels: List[str] = []
        token_confidences: List[float] = []
        for sw_range in token_sw_ranges:
            if sw_range is None:
                token_labels.append("O")
                token_confidences.append(0.0)
            else:
                sw_s, sw_e = sw_range
                sw_e = min(sw_e, total_sw)
                if sw_e <= sw_s:
                    token_labels.append("O")
                    token_confidences.append(0.0)
                else:
                    # Use first subword's label (BIO convention), avg confidence
                    token_labels.append(merged_labels[sw_s])
                    token_confidences.append(float(np.mean(merged_confs[sw_s:sw_e])))

        return token_labels, token_confidences

    def predict(self, text: str) -> List[Dict]:
        """Run NER prediction on input text.

        Automatically applies sliding window for texts longer than PhoBERT's
        256-token limit. For short texts, runs a single forward pass.

        Args:
            text: Raw input text (e.g., OCR output from a document).

        Returns:
            List of raw BIO predictions, each with:
                - type: Entity type (e.g., 'PATIENT_NAME')
                - text: Extracted text
                - start: Character start offset
                - end: Character end offset
                - label: Full BIO label (e.g., 'B-patient_name')
                - confidence: Prediction confidence score
        """
        tokens_with_pos = self.tokenize_with_offsets(text)
        if not tokens_with_pos:
            return []

        tokens = [t for t, _, _ in tokens_with_pos]

        # Encode all tokens to subwords
        subword_ids, token_sw_ranges = self._encode_tokens(tokens)

        # Run sliding window inference (handles both short and long texts)
        token_labels, token_confidences = self._predict_with_sliding_window(
            subword_ids, token_sw_ranges
        )

        # Decode BIO tags into entity spans
        raw_entities = merge_subtoken_predictions(
            tokens=tokens,
            labels=token_labels,
            confidences=token_confidences,
            tokens_with_pos=tokens_with_pos,
        )

        return raw_entities

    def predict_batch(self, texts: List[str]) -> List[List[Dict]]:
        """Run NER prediction on a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of entity lists, one per input text.
        """
        return [self.predict(text) for text in texts]
