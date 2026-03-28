"""
DocTypeDetector — Auto-detect Vietnamese document type from raw OCR text.

Keywords are loaded DYNAMICALLY from template files at runtime — no hardcoding.

Strategy:
  - Strong keywords: title lines in templates (lines without {{ }}, not boilerplate)
  - Weak keywords:   label prefixes before {{ }} on each line (e.g. "Họ và tên:")
  - Exclude keywords: strong keywords of OTHER doc types (cross-exclusion)

Template source: templates/{doc_type}/template_*.txt
Registry source: registry/documents.yaml (or HuggingFace Hub model card metadata)

The detector can be loaded from:
  1. Local repo path (development)
  2. HuggingFace Hub — downloads detector_rules.json from model repo
  3. Fallback: minimal keyword set from schema field names only
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Lines that appear in ALL Vietnamese documents — not useful for detection
_BOILERPLATE_LINES = {
    "cộng hòa xã hội chủ nghĩa việt nam",
    "socialist republic of viet nam",
    "độc lập - tự do - hạnh phúc",
    "independence - freedom - happiness",
    "độc lập tự do hạnh phúc",
}

# Scoring weights
_STRONG_WEIGHT = 3.0
_WEAK_WEIGHT = 1.0
_EXCLUDE_PENALTY = -10.0

# Default confidence threshold
_DEFAULT_THRESHOLD = 0.40


@dataclass
class DetectionResult:
    """Result of document type detection.

    Attributes:
        doc_type: Detected document type (e.g., 'cccd', 'giay_ra_vien').
            None if confidence is below threshold.
        confidence: Detection confidence score in [0.0, 1.0].
        scores: Raw normalized scores for all candidate doc types.
        is_confident: True if confidence >= threshold.
    """
    doc_type: Optional[str]
    confidence: float
    scores: Dict[str, float]
    is_confident: bool


@dataclass
class DocTypeRule:
    """Keyword rules for a document type.

    Attributes:
        strong_keywords: Title/header lines exclusive to this doc type.
        weak_keywords: Field label prefixes (e.g. "Họ và tên:").
        exclude_keywords: Keywords that indicate a DIFFERENT doc type.
    """
    strong_keywords: List[str] = field(default_factory=list)
    weak_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "strong_keywords": self.strong_keywords,
            "weak_keywords": self.weak_keywords,
            "exclude_keywords": self.exclude_keywords,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, List[str]]) -> "DocTypeRule":
        return cls(
            strong_keywords=d.get("strong_keywords", []),
            weak_keywords=d.get("weak_keywords", []),
            exclude_keywords=d.get("exclude_keywords", []),
        )


# ---------------------------------------------------------------------------
# Template parsing — extract keywords from template text files
# ---------------------------------------------------------------------------

def _extract_keywords_from_template(template_text: str) -> Tuple[List[str], List[str]]:
    """Parse a template file and extract strong and weak keywords.

    Strong keywords: lines that have NO {{ }} placeholders and are not boilerplate.
    Weak keywords: text before {{ }} on lines that have a placeholder.

    Args:
        template_text: Raw content of a template_*.txt file.

    Returns:
        Tuple of (strong_keywords, weak_keywords), all lowercased and stripped.
    """
    strong: List[str] = []
    weak: List[str] = []

    for raw_line in template_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line_lower = line.lower()

        if "{{" in line:
            # Extract text before the first {{ as a weak keyword (field label)
            prefix = re.split(r"\{\{", line)[0].strip()
            # Clean trailing punctuation like ":", "/", "-"
            prefix = re.sub(r"[\s:\/\-]+$", "", prefix).strip()
            if prefix and len(prefix) >= 3:
                weak.append(prefix.lower())
        else:
            # Pure text line — candidate for strong keyword
            # Skip boilerplate and very short lines
            if line_lower not in _BOILERPLATE_LINES and len(line) >= 4:
                strong.append(line_lower)

    return strong, weak


def _build_rules_from_templates(templates_dir: Path) -> Dict[str, DocTypeRule]:
    """Build detection rules by scanning all template files in a directory.

    Scans all template_*.txt files, extracts keywords, then applies
    cross-exclusion: strong keywords of doc type A become exclude keywords
    for all other doc types.

    Args:
        templates_dir: Path to the templates/ directory containing
            subdirectories per doc type.

    Returns:
        Dict mapping doc_type -> DocTypeRule.
    """
    raw_rules: Dict[str, Tuple[List[str], List[str]]] = {}

    for doc_dir in sorted(templates_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_type = doc_dir.name

        all_strong: List[str] = []
        all_weak: List[str] = []

        for tmpl_file in sorted(doc_dir.glob("template_*.txt")):
            content = tmpl_file.read_text(encoding="utf-8")
            s, w = _extract_keywords_from_template(content)
            all_strong.extend(s)
            all_weak.extend(w)

        if not all_strong and not all_weak:
            continue

        # Deduplicate while preserving order
        seen: set = set()
        dedup_strong = []
        for kw in all_strong:
            if kw not in seen:
                seen.add(kw)
                dedup_strong.append(kw)

        seen = set()
        dedup_weak = []
        for kw in all_weak:
            if kw not in seen:
                seen.add(kw)
                dedup_weak.append(kw)

        raw_rules[doc_type] = (dedup_strong, dedup_weak)

    # Cross-exclusion: strong keywords of others become exclude keywords
    rules: Dict[str, DocTypeRule] = {}
    for doc_type, (strong, weak) in raw_rules.items():
        exclude: List[str] = []
        for other_type, (other_strong, _) in raw_rules.items():
            if other_type != doc_type:
                exclude.extend(other_strong)
        rules[doc_type] = DocTypeRule(
            strong_keywords=strong,
            weak_keywords=weak,
            exclude_keywords=exclude,
        )

    return rules


def _build_rules_from_schema_only(templates_dir: Path) -> Dict[str, DocTypeRule]:
    """Minimal fallback: build rules from schema field names only.

    Used when no template_*.txt files are available (e.g., SDK installed
    without the full repo). Generates weak keywords from entity names.

    Args:
        templates_dir: Path to templates/ directory.

    Returns:
        Dict mapping doc_type -> DocTypeRule (weak keywords only).
    """
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


def _load_rules_from_hub(repo_id: str) -> Optional[Dict[str, DocTypeRule]]:
    """Download detector_rules.json from a HuggingFace Hub repo.

    The training pipeline saves this file alongside the model so that
    the SDK can load detection rules without the full repo.

    Args:
        repo_id: HuggingFace Hub repo ID (e.g., 'ngocthanhdoan/phobert-cccd-ner').

    Returns:
        Dict of rules, or None if the file is not available.
    """
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id=repo_id, filename="detector_rules.json")
        with open(local_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            doc_type: DocTypeRule.from_dict(rule_dict)
            for doc_type, rule_dict in raw.items()
        }
    except Exception:
        return None


def build_and_save_rules(templates_dir: Path, output_path: Path) -> Dict[str, DocTypeRule]:
    """Build rules from templates and save to detector_rules.json.

    Called during dataset generation / training to persist rules alongside
    the model on HuggingFace Hub.

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
    return rules


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize OCR text for keyword matching."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[_\-]{2,}", " ", text)
    return text.strip()


def _score_text(text_norm: str, rule: DocTypeRule) -> float:
    """Compute raw score for a document type given normalized text."""
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

    Keywords are loaded DYNAMICALLY from template files — not hardcoded.
    The detector can be initialized from:

    1. **Local templates directory** (full repo):
       ``DocTypeDetector.from_templates("/path/to/VietNerm/templates")``

    2. **Saved rules file** (detector_rules.json):
       ``DocTypeDetector.from_rules_file("/path/to/detector_rules.json")``

    3. **HuggingFace Hub** (SDK installed via pip, no local repo):
       ``DocTypeDetector.from_hub("ngocthanhdoan/phobert-cccd-ner")``

    4. **Auto-discover** (tries all sources in order):
       ``DocTypeDetector()``  — searches for templates/ relative to cwd

    Args:
        rules: Pre-built rules dict. If None, auto-discover is attempted.
        threshold: Minimum confidence to accept a detection (default 0.40).

    Example::

        >>> from vietnerm.detector import DocTypeDetector
        >>> detector = DocTypeDetector.from_templates("templates/")
        >>> result = detector.detect("CĂN CƯỚC CÔNG DÂN\\nSố: 079203030140")
        >>> print(result.doc_type, result.confidence)
        cccd 1.0
    """

    def __init__(
        self,
        rules: Optional[Dict[str, DocTypeRule]] = None,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.rules: Dict[str, DocTypeRule] = rules or {}

        # Auto-discover if no rules provided
        if not self.rules:
            self.rules = self._auto_discover_rules()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_templates(cls, templates_dir: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Build detector from local template files.

        Args:
            templates_dir: Path to templates/ directory.
            threshold: Detection confidence threshold.
        """
        rules = _build_rules_from_templates(Path(templates_dir))
        return cls(rules=rules, threshold=threshold)

    @classmethod
    def from_rules_file(cls, rules_path: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Load detector from a saved detector_rules.json file.

        Args:
            rules_path: Path to detector_rules.json.
            threshold: Detection confidence threshold.
        """
        with open(rules_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rules = {dt: DocTypeRule.from_dict(d) for dt, d in raw.items()}
        return cls(rules=rules, threshold=threshold)

    @classmethod
    def from_hub(cls, repo_id: str, threshold: float = _DEFAULT_THRESHOLD) -> "DocTypeDetector":
        """Load detector rules from HuggingFace Hub (detector_rules.json).

        Args:
            repo_id: HuggingFace Hub repo ID.
            threshold: Detection confidence threshold.
        """
        rules = _load_rules_from_hub(repo_id)
        if not rules:
            raise ValueError(
                f"detector_rules.json not found in HuggingFace Hub repo: {repo_id}"
            )
        return cls(rules=rules, threshold=threshold)

    # ------------------------------------------------------------------
    # Auto-discover
    # ------------------------------------------------------------------

    def _auto_discover_rules(self) -> Dict[str, DocTypeRule]:
        """Try to discover rules from common locations.

        Search order:
          1. templates/ relative to current working directory
          2. detector_rules.json in current directory
          3. Schema-only fallback (field names as weak keywords)
        """
        # 1. templates/ in cwd
        for candidate in [
            Path("templates"),
            Path(__file__).parent.parent.parent / "templates",  # repo root
        ]:
            if candidate.is_dir() and any(candidate.glob("*/template_*.txt")):
                return _build_rules_from_templates(candidate)

        # 2. detector_rules.json in cwd
        rules_file = Path("detector_rules.json")
        if rules_file.exists():
            with open(rules_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {dt: DocTypeRule.from_dict(d) for dt, d in raw.items()}

        # 3. Schema-only fallback
        for candidate in [
            Path("templates"),
            Path(__file__).parent.parent.parent / "templates",
        ]:
            if candidate.is_dir() and any(candidate.glob("*/schema.yaml")):
                return _build_rules_from_schema_only(candidate)

        return {}

    # ------------------------------------------------------------------
    # Runtime registration
    # ------------------------------------------------------------------

    def register(self, doc_type: str, rule: DocTypeRule) -> None:
        """Register a new document type rule at runtime.

        Also updates cross-exclusion: the new type's strong keywords are
        added as exclude keywords for all existing types, and vice versa.

        Args:
            doc_type: Document type identifier (e.g., 'hop_dong').
            rule: DocTypeRule with keywords for this doc type.
        """
        # Add new type's strong keywords as exclude for existing types
        for existing_type, existing_rule in self.rules.items():
            existing_rule.exclude_keywords.extend(rule.strong_keywords)
            rule.exclude_keywords.extend(existing_rule.strong_keywords)

        self.rules[doc_type] = rule

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> DetectionResult:
        """Detect document type from raw OCR text.

        Args:
            text: Raw OCR text from a Vietnamese document.

        Returns:
            DetectionResult with doc_type, confidence, and per-type scores.
        """
        if not self.rules:
            return DetectionResult(
                doc_type=None, confidence=0.0, scores={}, is_confident=False
            )

        text_norm = _normalize_text(text)
        raw_scores: Dict[str, float] = {
            dt: _score_text(text_norm, rule)
            for dt, rule in self.rules.items()
        }

        total = sum(raw_scores.values())
        if total == 0:
            return DetectionResult(
                doc_type=None,
                confidence=0.0,
                scores={k: 0.0 for k in raw_scores},
                is_confident=False,
            )

        norm_scores = {k: v / total for k, v in raw_scores.items()}
        best_type = max(norm_scores, key=norm_scores.__getitem__)
        best_conf = norm_scores[best_type]

        return DetectionResult(
            doc_type=best_type if best_conf >= self.threshold else None,
            confidence=round(best_conf, 4),
            scores={k: round(v, 4) for k, v in norm_scores.items()},
            is_confident=best_conf >= self.threshold,
        )

    def detect_top_n(self, text: str, n: int = 3) -> List[Tuple[str, float]]:
        """Return top-N candidate doc types with confidence scores.

        Args:
            text: Raw OCR text.
            n: Number of top candidates to return.

        Returns:
            List of (doc_type, confidence) tuples, sorted by confidence desc.
        """
        result = self.detect(text)
        return sorted(result.scores.items(), key=lambda x: x[1], reverse=True)[:n]

    def explain(self, text: str) -> Dict[str, Any]:
        """Explain which keywords matched for each doc type.

        Useful for debugging detection results.

        Args:
            text: Raw OCR text.

        Returns:
            Dict mapping doc_type to matched keywords and score breakdown.
        """
        text_norm = _normalize_text(text)
        explanation: Dict[str, Any] = {}

        for doc_type, rule in self.rules.items():
            matched_strong = [kw for kw in rule.strong_keywords if kw in text_norm]
            matched_weak = [kw for kw in rule.weak_keywords if kw in text_norm]
            matched_exclude = [kw for kw in rule.exclude_keywords if kw in text_norm]
            score = (
                len(matched_strong) * _STRONG_WEIGHT
                + len(matched_weak) * _WEAK_WEIGHT
                + len(matched_exclude) * _EXCLUDE_PENALTY
            )
            explanation[doc_type] = {
                "score": max(score, 0.0),
                "matched_strong": matched_strong,
                "matched_weak": matched_weak,
                "matched_exclude": matched_exclude,
            }

        return explanation
