"""
DocTypeDetector — Auto-detect Vietnamese document type from raw OCR text.

Detection strategy (in priority order):
  1. **TF-IDF + FAISS** (best accuracy): Build a TF-IDF vector index from all
     template text. At runtime, query text is vectorized and compared against
     doc-type vectors using cosine similarity. Automatically built from templates/
     and saved as detector_index.bin + detector_vocab.json alongside the model.

  2. **Keyword scoring** (fast fallback): If FAISS index is not available,
     falls back to weighted keyword matching (strong/weak/exclude keywords
     extracted dynamically from template files).

  3. **Schema-only** (minimal fallback): If no templates are available (SDK
     installed without full repo), uses entity field names as weak keywords.

Loading sources:
  1. Local templates/ directory (development / training pipeline)
  2. HuggingFace Hub — downloads detector_index.bin + detector_vocab.json
  3. Saved detector_rules.json (keyword fallback)
  4. Auto-discover from cwd

Template source: templates/{doc_type}/template_*.txt
Registry source: registry/documents.yaml
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOILERPLATE_LINES = {
    "cộng hòa xã hội chủ nghĩa việt nam",
    "socialist republic of viet nam",
    "độc lập - tự do - hạnh phúc",
    "independence - freedom - happiness",
    "độc lập tự do hạnh phúc",
}

_STRONG_WEIGHT = 3.0
_WEAK_WEIGHT = 1.0
_EXCLUDE_PENALTY = -10.0
_DEFAULT_THRESHOLD = 0.35

# Minimum absolute TF-IDF cosine similarity for the best doc type
_TFIDF_MIN_SCORE = 0.10
# Minimum ratio of best_score / second_best_score to accept detection
# (1.08 = best must be at least 8% higher than second best)
_TFIDF_MIN_MARGIN = 1.08


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result of document type detection.

    Attributes:
        doc_type: Detected document type (e.g., 'cccd', 'giay_ra_vien').
            None if confidence is below threshold.
        confidence: Detection confidence score in [0.0, 1.0].
        scores: Raw normalized scores for all candidate doc types.
        is_confident: True if confidence >= threshold.
        method: Detection method used ('tfidf', 'keyword', 'none').
    """
    doc_type: Optional[str]
    confidence: float
    scores: Dict[str, float]
    is_confident: bool
    method: str = "keyword"


@dataclass
class DocTypeRule:
    """Keyword rules for a document type (used as fallback when FAISS unavailable).

    Attributes:
        strong_keywords: Title/header lines exclusive to this doc type.
        weak_keywords: Field label prefixes (e.g. "Họ và tên:").
        exclude_keywords: Keywords that indicate a DIFFERENT doc type.
        corpus: Full template text for TF-IDF vectorization.
    """
    strong_keywords: List[str] = field(default_factory=list)
    weak_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    corpus: str = ""

    def to_dict(self) -> Dict:
        return {
            "strong_keywords": self.strong_keywords,
            "weak_keywords": self.weak_keywords,
            "exclude_keywords": self.exclude_keywords,
            "corpus": self.corpus,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "DocTypeRule":
        return cls(
            strong_keywords=d.get("strong_keywords", []),
            weak_keywords=d.get("weak_keywords", []),
            exclude_keywords=d.get("exclude_keywords", []),
            corpus=d.get("corpus", ""),
        )


# ---------------------------------------------------------------------------
# Template parsing
# ---------------------------------------------------------------------------

def _extract_keywords_from_template(template_text: str) -> Tuple[List[str], List[str]]:
    """Parse a template file and extract strong and weak keywords."""
    strong: List[str] = []
    weak: List[str] = []

    for raw_line in template_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line_lower = line.lower()

        if "{{" in line:
            prefix = re.split(r"\{\{", line)[0].strip()
            prefix = re.sub(r"[\s:\/\-]+$", "", prefix).strip()
            if prefix and len(prefix) >= 3:
                weak.append(prefix.lower())
        else:
            if line_lower not in _BOILERPLATE_LINES and len(line) >= 4:
                strong.append(line_lower)

    return strong, weak


def _build_corpus_from_templates(doc_dir: Path) -> str:
    """Build a single text corpus from all template files of a doc type.

    Keeps both the surrounding text AND the placeholder names (e.g. 'full_name',
    'license_class') since these are highly discriminative for each doc type.
    Also includes schema field names for additional signal.
    """
    parts = []

    # Template files — keep placeholder names as words
    for tmpl_file in sorted(doc_dir.glob("template_*.txt")):
        content = tmpl_file.read_text(encoding="utf-8")
        # Replace {{ field_name }} with the field name itself (underscores → spaces)
        expanded = re.sub(
            r"\{\{\s*([^}]+?)\s*\}\}",
            lambda m: m.group(1).strip().replace("_", " "),
            content,
        )
        cleaned = re.sub(r"\s+", " ", expanded).strip()
        parts.append(cleaned)

    # Schema file — add entity names and labels as additional signal
    schema_file = doc_dir / "schema.yaml"
    if schema_file.exists():
        try:
            import yaml
            data = yaml.safe_load(schema_file.read_text(encoding="utf-8"))
            entities = data.get("entities", [])
            for ent in entities:
                name = ent.get("name", "").replace("_", " ")
                label = ent.get("label", "").replace("_", " ")
                if name:
                    parts.append(f"{name} {label}" * 3)  # boost entity names
        except Exception:
            pass

    return " ".join(parts)


def _build_rules_from_templates(templates_dir: Path) -> Dict[str, "DocTypeRule"]:
    """Build detection rules by scanning all template files."""
    raw_rules: Dict[str, Tuple[List[str], List[str], str]] = {}

    for doc_dir in sorted(templates_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_type = doc_dir.name

        all_strong: List[str] = []
        all_weak: List[str] = []
        corpus = _build_corpus_from_templates(doc_dir)

        for tmpl_file in sorted(doc_dir.glob("template_*.txt")):
            content = tmpl_file.read_text(encoding="utf-8")
            s, w = _extract_keywords_from_template(content)
            all_strong.extend(s)
            all_weak.extend(w)

        if not all_strong and not all_weak and not corpus:
            continue

        # Deduplicate
        seen: set = set()
        dedup_strong = [kw for kw in all_strong if not (kw in seen or seen.add(kw))]
        seen = set()
        dedup_weak = [kw for kw in all_weak if not (kw in seen or seen.add(kw))]

        raw_rules[doc_type] = (dedup_strong, dedup_weak, corpus)

    # Cross-exclusion
    rules: Dict[str, DocTypeRule] = {}
    for doc_type, (strong, weak, corpus) in raw_rules.items():
        exclude: List[str] = []
        for other_type, (other_strong, _, _) in raw_rules.items():
            if other_type != doc_type:
                exclude.extend(other_strong)
        rules[doc_type] = DocTypeRule(
            strong_keywords=strong,
            weak_keywords=weak,
            exclude_keywords=exclude,
            corpus=corpus,
        )

    return rules


def _build_rules_from_schema_only(templates_dir: Path) -> Dict[str, "DocTypeRule"]:
    """Minimal fallback: build rules from schema field names only."""
    import yaml
    rules: Dict[str, DocTypeRule] = {}
    for schema_file in sorted(templates_dir.glob("*/schema.yaml")):
        try:
            data = yaml.safe_load(schema_file.read_text(encoding="utf-8"))
            doc_type = data.get("doc_type", schema_file.parent.name)
            entities = data.get("entities", [])
            weak = [e["name"].replace("_", " ") for e in entities if "name" in e]
            rules[doc_type] = DocTypeRule(weak_keywords=weak)
        except Exception:
            continue
    return rules


# ---------------------------------------------------------------------------
# TF-IDF + FAISS index
# ---------------------------------------------------------------------------

class _TFIDFIndex:
    """TF-IDF vectorizer + FAISS cosine similarity index for doc type detection.

    Each doc type is represented by a single TF-IDF document vector built
    from its full template corpus. At query time, the input text is vectorized
    using the SAME fitted vectorizer and compared against doc type vectors
    using cosine similarity.

    For small numbers of doc types (< 100), FAISS IndexFlatIP is used directly.
    For larger collections, upgrade to IndexIVFFlat.
    """

    def __init__(
        self,
        doc_types: List[str],
        vectors: np.ndarray,
        vocab: Dict[str, int],
        idf: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            doc_types: Ordered list of doc type names.
            vectors: Float32 array of shape (n_doc_types, n_features), L2-normalized.
            vocab: TF-IDF vocabulary mapping term -> index.
            idf: IDF weights for each vocabulary term (for query vectorization).
        """
        self.doc_types = doc_types
        self.vectors = vectors  # (n, d) float32, L2-normalized
        self.vocab = vocab
        self.idf = np.array(idf, dtype=np.float32) if idf is not None else None
        self._index = None
        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        try:
            import faiss
            d = self.vectors.shape[1]
            index = faiss.IndexFlatIP(d)  # inner product = cosine (after L2 norm)
            index.add(self.vectors.astype(np.float32))
            self._index = index
        except ImportError:
            self._index = None  # fallback to numpy dot product

    def _vectorize(self, text: str) -> np.ndarray:
        """Convert text to a normalized TF-IDF vector using fitted IDF weights."""
        from collections import Counter
        import math

        text_norm = re.sub(r"\s+", " ", text.lower()).strip()
        terms = text_norm.split()
        if not terms:
            return np.zeros(len(self.vocab), dtype=np.float32)

        tf = Counter(terms)
        vec = np.zeros(len(self.vocab), dtype=np.float32)

        if self.idf is not None:
            # Use fitted IDF weights (same as sklearn TfidfVectorizer with sublinear_tf)
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    tf_val = 1 + math.log(count) if count > 0 else 0  # sublinear TF
                    vec[idx] = tf_val * float(self.idf[idx])
        else:
            # Fallback: plain TF
            n_terms = len(terms)
            for term, count in tf.items():
                if term in self.vocab:
                    vec[self.vocab[term]] = count / n_terms

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def query(self, text: str) -> Tuple[List[str], List[float]]:
        """Query the index for the most similar doc types.

        Returns:
            Tuple of (doc_types_ordered, cosine_scores_ordered) sorted desc.
        """
        vec = self._vectorize(text).reshape(1, -1).astype(np.float32)

        if self._index is not None:
            import faiss
            scores, indices = self._index.search(vec, len(self.doc_types))
            ordered_types = [self.doc_types[i] for i in indices[0]]
            ordered_scores = [float(s) for s in scores[0]]
        else:
            # Fallback: numpy dot product
            scores = (self.vectors @ vec.T).flatten()
            order = np.argsort(scores)[::-1]
            ordered_types = [self.doc_types[i] for i in order]
            ordered_scores = [float(scores[i]) for i in order]

        return ordered_types, ordered_scores

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dict."""
        vocab_serializable = {k: int(v) for k, v in self.vocab.items()}
        result = {
            "doc_types": self.doc_types,
            "vectors": self.vectors.tolist(),
            "vocab": vocab_serializable,
        }
        if self.idf is not None:
            result["idf"] = self.idf.tolist()
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "_TFIDFIndex":
        return cls(
            doc_types=d["doc_types"],
            vectors=np.array(d["vectors"], dtype=np.float32),
            vocab=d["vocab"],
            idf=d.get("idf"),
        )

    @classmethod
    def build_from_rules(cls, rules: Dict[str, "DocTypeRule"]) -> Optional["_TFIDFIndex"]:
        """Build a TF-IDF index from DocTypeRule corpus texts.

        Returns None if scikit-learn is not available or corpora are empty.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            return None

        doc_types = sorted(rules.keys())
        corpora = [rules[dt].corpus for dt in doc_types]

        if not any(corpora):
            return None

        try:
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=8000,
                sublinear_tf=True,
            )
            tfidf_matrix = vectorizer.fit_transform(corpora).toarray().astype(np.float32)

            # L2 normalize each row
            norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            tfidf_matrix /= norms

            vocab = vectorizer.vocabulary_
            idf = vectorizer.idf_.tolist()  # save IDF weights for query vectorization

            return cls(doc_types=doc_types, vectors=tfidf_matrix, vocab=vocab, idf=idf)
        except Exception:
            return None


def build_and_save_index(rules: Dict[str, "DocTypeRule"], output_dir: Path) -> Optional["_TFIDFIndex"]:
    """Build TF-IDF index from rules and save to output_dir/detector_index.json.

    Called during training pipeline to persist the index alongside the model
    on HuggingFace Hub.

    Args:
        rules: DocTypeRule dict (must have corpus text populated).
        output_dir: Directory to write detector_index.json.

    Returns:
        The built _TFIDFIndex, or None if build failed.
    """
    index = _TFIDFIndex.build_from_rules(rules)
    if index is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "detector_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, ensure_ascii=False)
    return index


def build_and_save_rules(templates_dir: Path, output_path: Path) -> Dict[str, "DocTypeRule"]:
    """Build rules from templates and save to detector_rules.json.

    Also builds and saves the TF-IDF index to the same directory.

    Args:
        templates_dir: Path to templates/ directory.
        output_path: Path to write detector_rules.json.

    Returns:
        The built rules dict.
    """
    rules = _build_rules_from_templates(templates_dir)
    serializable = {dt: rule.to_dict() for dt, rule in rules.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # Also build and save TF-IDF index
    build_and_save_index(rules, output_path.parent)

    return rules


# ---------------------------------------------------------------------------
# Hub loading
# ---------------------------------------------------------------------------

def _load_rules_from_hub(repo_id: str) -> Optional[Dict[str, "DocTypeRule"]]:
    """Download detector_rules.json from a HuggingFace Hub repo."""
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_id, filename="detector_rules.json")
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {dt: DocTypeRule.from_dict(d) for dt, d in raw.items()}
    except Exception:
        return None


def _load_index_from_hub(repo_id: str) -> Optional["_TFIDFIndex"]:
    """Download detector_index.json from a HuggingFace Hub repo."""
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_id, filename="detector_index.json")
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _TFIDFIndex.from_dict(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Scoring (keyword fallback)
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[_\-]{2,}", " ", text)
    return text.strip()


def _score_text(text_norm: str, rule: "DocTypeRule") -> float:
    score = 0.0
    for kw in rule.strong_keywords:
        if kw in text_norm:
            score += _STRONG_WEIGHT
    for kw in rule.weak_keywords:
        if kw in text_norm:
            score += _WEAK_WEIGHT
    for kw in rule.exclude_keywords:
        if kw in text_norm:
            score += _EXCLUDE_PENALTY
    return max(score, 0.0)


# ---------------------------------------------------------------------------
# DocTypeDetector
# ---------------------------------------------------------------------------

class DocTypeDetector:
    """Detect Vietnamese document type from raw OCR text.

    Uses TF-IDF + FAISS cosine similarity as primary method (built from
    template corpora), with keyword scoring as fallback.

    Loading sources (in priority order):
      1. Local templates/ directory → builds TF-IDF index on-the-fly
      2. HuggingFace Hub → downloads pre-built index + rules
      3. Saved detector_rules.json → keyword fallback only
      4. Auto-discover from cwd

    Example::

        # Development (full repo)
        detector = DocTypeDetector.from_templates("templates/")
        result = detector.detect("CĂN CƯỚC CÔNG DÂN\\nSố: 079203030140")
        print(result.doc_type, result.confidence, result.method)
        # cccd 0.9231 tfidf

        # Production (SDK via pip, model on HF Hub)
        detector = DocTypeDetector.from_hub("ngocthanhdoan/phobert-cccd-ner")
        result = detector.detect(ocr_text)
    """

    def __init__(
        self,
        rules: Optional[Dict[str, DocTypeRule]] = None,
        index: Optional["_TFIDFIndex"] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.rules: Dict[str, DocTypeRule] = rules or {}
        self._index: Optional[_TFIDFIndex] = index

        # Auto-discover if nothing provided
        if not self.rules and self._index is None:
            self.rules, self._index = self._auto_discover()

        # If we have rules but no index, try to build index from corpora
        if self._index is None and self.rules:
            self._index = _TFIDFIndex.build_from_rules(self.rules)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_templates(cls, templates_dir: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Build detector from local template files.

        Builds both keyword rules and TF-IDF index from template corpora.
        """
        rules = _build_rules_from_templates(Path(templates_dir))
        index = _TFIDFIndex.build_from_rules(rules)
        return cls(rules=rules, index=index, threshold=threshold)

    @classmethod
    def from_rules_file(cls, rules_path: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Load detector from a saved detector_rules.json file (keyword fallback)."""
        with open(rules_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rules = {dt: DocTypeRule.from_dict(d) for dt, d in raw.items()}
        # Try to load index from same directory
        index_path = Path(rules_path).parent / "detector_index.json"
        index = None
        if index_path.exists():
            with open(index_path, "r", encoding="utf-8") as f:
                index = _TFIDFIndex.from_dict(json.load(f))
        if index is None:
            index = _TFIDFIndex.build_from_rules(rules)
        return cls(rules=rules, index=index, threshold=threshold)

    @classmethod
    def from_hub(cls, repo_id: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Load detector from HuggingFace Hub.

        Downloads detector_index.json (TF-IDF) and detector_rules.json (fallback).
        """
        index = _load_index_from_hub(repo_id)
        rules = _load_rules_from_hub(repo_id)
        if not rules and index is None:
            raise ValueError(
                f"Neither detector_index.json nor detector_rules.json found in: {repo_id}"
            )
        return cls(rules=rules or {}, index=index, threshold=threshold)

    # ------------------------------------------------------------------
    # Auto-discover
    # ------------------------------------------------------------------

    def _auto_discover(self) -> Tuple[Dict[str, DocTypeRule], Optional["_TFIDFIndex"]]:
        """Try to discover rules from common locations."""
        # 1. templates/ in cwd or repo root
        for candidate in [
            Path("templates"),
            Path(__file__).parent.parent.parent / "templates",
        ]:
            if candidate.is_dir() and any(candidate.glob("*/template_*.txt")):
                rules = _build_rules_from_templates(candidate)
                index = _TFIDFIndex.build_from_rules(rules)
                return rules, index

        # 2. detector_rules.json + detector_index.json in cwd
        rules_file = Path("detector_rules.json")
        if rules_file.exists():
            with open(rules_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            rules = {dt: DocTypeRule.from_dict(d) for dt, d in raw.items()}
            index_file = Path("detector_index.json")
            index = None
            if index_file.exists():
                with open(index_file, "r", encoding="utf-8") as f:
                    index = _TFIDFIndex.from_dict(json.load(f))
            if index is None:
                index = _TFIDFIndex.build_from_rules(rules)
            return rules, index

        # 3. Schema-only fallback
        for candidate in [
            Path("templates"),
            Path(__file__).parent.parent.parent / "templates",
        ]:
            if candidate.is_dir() and any(candidate.glob("*/schema.yaml")):
                rules = _build_rules_from_schema_only(candidate)
                return rules, None

        return {}, None

    # ------------------------------------------------------------------
    # Runtime registration
    # ------------------------------------------------------------------

    def register(self, doc_type: str, rule: DocTypeRule) -> None:
        """Register a new document type rule at runtime.

        Rebuilds the TF-IDF index to include the new doc type.
        """
        for existing_rule in self.rules.values():
            existing_rule.exclude_keywords.extend(rule.strong_keywords)
            rule.exclude_keywords.extend(existing_rule.strong_keywords)

        self.rules[doc_type] = rule
        # Rebuild index to include new doc type
        self._index = _TFIDFIndex.build_from_rules(self.rules)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> DetectionResult:
        """Detect document type from raw OCR text.

        Uses TF-IDF + FAISS if available, falls back to keyword scoring.

        Args:
            text: Raw OCR text from a Vietnamese document.

        Returns:
            DetectionResult with doc_type, confidence, scores, and method.
        """
        if not self.rules and self._index is None:
            return DetectionResult(
                doc_type=None, confidence=0.0, scores={}, is_confident=False, method="none"
            )

        # --- TF-IDF + FAISS (primary) ---
        if self._index is not None:
            ordered_types, ordered_scores = self._index.query(text)

            best_raw_score = ordered_scores[0] if ordered_scores else 0.0

            # Require minimum absolute cosine similarity to avoid false positives
            # on unrelated text (which still gets normalized to high relative score)
            second_raw = ordered_scores[1] if len(ordered_scores) > 1 else 0.0
            margin_ok = (second_raw <= 0) or (best_raw_score / (second_raw + 1e-9) >= _TFIDF_MIN_MARGIN)

            if best_raw_score < _TFIDF_MIN_SCORE or not margin_ok:
                return DetectionResult(
                    doc_type=None,
                    confidence=0.0,
                    scores={t: 0.0 for t in ordered_types},
                    is_confident=False,
                    method="tfidf",
                )

            # Softmax-normalize positive scores for interpretable confidence
            scores_arr = np.array(ordered_scores, dtype=np.float64)
            scores_arr = np.clip(scores_arr, 0, None)
            total = scores_arr.sum()

            if total > 0:
                norm_scores = scores_arr / total
                scores_dict = {t: round(float(s), 4) for t, s in zip(ordered_types, norm_scores)}
                best_type = ordered_types[0]
                best_conf = float(norm_scores[0])

                if best_conf >= self.threshold:
                    return DetectionResult(
                        doc_type=best_type,
                        confidence=round(best_conf, 4),
                        scores=scores_dict,
                        is_confident=True,
                        method="tfidf",
                    )
                # Low confidence — still return result but mark not confident
                return DetectionResult(
                    doc_type=None,
                    confidence=round(best_conf, 4),
                    scores=scores_dict,
                    is_confident=False,
                    method="tfidf",
                )

        # --- Keyword scoring (fallback) ---
        if self.rules:
            text_norm = _normalize_text(text)
            raw_scores = {dt: _score_text(text_norm, rule) for dt, rule in self.rules.items()}
            total = sum(raw_scores.values())

            if total > 0:
                norm_scores = {k: v / total for k, v in raw_scores.items()}
                best_type = max(norm_scores, key=norm_scores.__getitem__)
                best_conf = norm_scores[best_type]
                return DetectionResult(
                    doc_type=best_type if best_conf >= self.threshold else None,
                    confidence=round(best_conf, 4),
                    scores={k: round(v, 4) for k, v in norm_scores.items()},
                    is_confident=best_conf >= self.threshold,
                    method="keyword",
                )

        return DetectionResult(
            doc_type=None, confidence=0.0,
            scores={dt: 0.0 for dt in self.rules},
            is_confident=False, method="none"
        )

    def detect_top_n(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """Return top-N candidate doc types with confidence scores."""
        result = self.detect(text)
        return sorted(result.scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def explain(self, text: str) -> Dict[str, Any]:
        """Explain detection result — shows TF-IDF scores and matched keywords.

        Args:
            text: Raw OCR text.

        Returns:
            Dict with 'method', 'tfidf_scores' (if available), and
            'keyword_matches' per doc type.
        """
        explanation: Dict[str, Any] = {"method": "none", "tfidf_scores": {}, "keyword_matches": {}}

        # TF-IDF scores
        if self._index is not None:
            ordered_types, ordered_scores = self._index.query(text)
            explanation["tfidf_scores"] = {t: round(float(s), 4) for t, s in zip(ordered_types, ordered_scores)}
            explanation["method"] = "tfidf"

        # Keyword matches
        if self.rules:
            text_norm = _normalize_text(text)
            for doc_type, rule in self.rules.items():
                matched_strong = [kw for kw in rule.strong_keywords if kw in text_norm]
                matched_weak = [kw for kw in rule.weak_keywords if kw in text_norm]
                matched_exclude = [kw for kw in rule.exclude_keywords if kw in text_norm]
                score = (
                    len(matched_strong) * _STRONG_WEIGHT
                    + len(matched_weak) * _WEAK_WEIGHT
                    + len(matched_exclude) * _EXCLUDE_PENALTY
                )
                explanation["keyword_matches"][doc_type] = {
                    "score": max(score, 0.0),
                    "matched_strong": matched_strong,
                    "matched_weak": matched_weak,
                    "matched_exclude": matched_exclude,
                }
            if not explanation["tfidf_scores"]:
                explanation["method"] = "keyword"

        return explanation
