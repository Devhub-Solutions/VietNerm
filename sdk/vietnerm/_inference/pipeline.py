"""
NER Pipeline - Load PhoBERT model and run NER prediction.

Supports loading from:
  - Local path: /path/to/model/
  - HuggingFace Hub: username/phobert-{doc}-ner
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .postprocess import merge_subtoken_predictions, clean_entity_boundaries, compute_confidence


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

    Args:
        model_path: Local path or HuggingFace hub ID for the model.
        device: Device to run inference on ('cpu', 'cuda', or 'auto').
        max_length: Maximum token sequence length.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length

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

    def predict(self, text: str) -> List[Dict]:
        """Run NER prediction on input text.

        Args:
            text: Raw input text (e.g., OCR output from a document).

        Returns:
            List of raw BIO predictions, each with:
                - type: Entity type (e.g., 'PATIENT_NAME')
                - text: Extracted text
                - start: Character start offset
                - end: Character end offset
                - label: Full BIO label (e.g., 'B-PATIENT_NAME_VALUE')
                - confidence: Prediction confidence score
        """
        tokens_with_pos = self.tokenize_with_offsets(text)
        if not tokens_with_pos:
            return []

        tokens = [t for t, _, _ in tokens_with_pos]

        # Build input_ids and track subword→token mapping
        input_ids: List[int] = [self.tokenizer.cls_token_id]
        token_to_subword_start: List[Optional[int]] = []
        token_to_subword_range: List[Optional[Tuple[int, int]]] = []

        for tok in tokens:
            word_tokens = self.tokenizer.encode(tok, add_special_tokens=False)
            if not word_tokens:
                token_to_subword_start.append(None)
                token_to_subword_range.append(None)
                continue
            sw_start = len(input_ids)
            token_to_subword_start.append(sw_start)
            token_to_subword_range.append((sw_start, sw_start + len(word_tokens)))
            input_ids.extend(word_tokens)

        input_ids.append(self.tokenizer.sep_token_id)
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        inputs = torch.tensor([input_ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(-1).tolist()
        max_probs = probs.max(-1).values.tolist()

        # Map predictions back to token level
        token_labels: List[str] = []
        token_confidences: List[float] = []
        for sw_start, sw_range in zip(token_to_subword_start, token_to_subword_range):
            if sw_start is None or sw_start >= len(pred_ids):
                token_labels.append("O")
                token_confidences.append(0.0)
            else:
                token_labels.append(self.id2label.get(pred_ids[sw_start], "O"))
                # Average confidence across subword tokens
                if sw_range is not None:
                    sw_s, sw_e = sw_range
                    sw_e = min(sw_e, len(max_probs))
                    conf = np.mean(max_probs[sw_s:sw_e]) if sw_e > sw_s else 0.0
                    token_confidences.append(float(conf))
                else:
                    token_confidences.append(float(max_probs[sw_start]))

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
