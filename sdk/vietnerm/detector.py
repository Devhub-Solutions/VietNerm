"""
DocTypeDetector — Auto-detect Vietnamese document type from raw OCR text.

Strategy: Keyword-based scoring with confidence threshold.
Each document type has a set of strong (exclusive) and weak (shared) keywords.
The detector scores each type and returns the best match with confidence.

Supports all doc types registered in the system.
New doc types can be added by extending KEYWORD_RULES.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DetectionResult:
    """Result of document type detection.

    Attributes:
        doc_type: Detected document type (e.g., 'cccd', 'giay_ra_vien').
            None if confidence is below threshold.
        confidence: Detection confidence score in [0.0, 1.0].
        scores: Raw scores for all candidate doc types.
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
        strong_keywords: Keywords that are highly exclusive to this doc type.
            Each match adds a large weight.
        weak_keywords: Keywords that may appear in multiple doc types.
            Each match adds a small weight.
        exclude_keywords: If any of these appear, this doc type is penalized.
    """
    strong_keywords: List[str] = field(default_factory=list)
    weak_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword rules per document type
# Derived from template text analysis + common OCR output patterns
# ---------------------------------------------------------------------------
KEYWORD_RULES: Dict[str, DocTypeRule] = {
    "cccd": DocTypeRule(
        strong_keywords=[
            "căn cước công dân",
            "cccd",
            "số định danh cá nhân",
            "căn cước",
            "có giá trị đến",
            "chứng minh nhân dân",
            "cmnd",
            # ASCII fallback for OCR without diacritics
            "can cuoc cong dan",
            "so dinh danh ca nhan",
            "co gia tri den",
            "chung minh nhan dan",
        ],
        weak_keywords=[
            "họ, chữ đệm và tên khai sinh",
            "họ và tên",
            "ngày tháng năm sinh",
            "quê quán",
            "nơi thường trú",
            "quốc tịch",
            "giới tính",
            # ASCII fallback
            "ho va ten",
            "ngay sinh",
            "que quan",
            "noi thuong tru",
            "quoc tich",
        ],
        exclude_keywords=[
            "giấy ra viện",
            "giấy đăng ký xe",
            "chẩn đoán",
            "biển số",
        ],
    ),

    "giay_ra_vien": DocTypeRule(
        strong_keywords=[
            "giấy ra viện",
            "ra viện",
            "vào viện",
            "chẩn đoán",
            "phương pháp điều trị",
            "bệnh viện",
            "khoa",
            "mã số bhxh",
            "bhyt",
            "người bệnh",
            "họ tên người bệnh",
        ],
        weak_keywords=[
            "năm sinh",
            "dân tộc",
            "nghề nghiệp",
            "địa chỉ",
            "giới tính",
            "ghi chú",
            "số lưu trữ",
        ],
        exclude_keywords=[
            "căn cước",
            "giấy đăng ký xe",
            "biển số",
            "số định danh",
        ],
    ),

    "vehicle_registration": DocTypeRule(
        strong_keywords=[
            "giấy đăng ký xe",
            "đăng ký xe",
            "biển số đăng ký",
            "biển số",
            "số máy",
            "số khung",
            "nhãn hiệu",
            "loại xe",
            "chủ xe",
            "màu sơn",
        ],
        weak_keywords=[
            "địa chỉ",
            "ngày đăng ký",
            "số loại",
        ],
        exclude_keywords=[
            "căn cước",
            "giấy ra viện",
            "chẩn đoán",
            "bệnh viện",
        ],
    ),
}

# Scoring weights
_STRONG_WEIGHT = 3.0
_WEAK_WEIGHT = 1.0
_EXCLUDE_PENALTY = -10.0

# Default confidence threshold
_DEFAULT_THRESHOLD = 0.40


def _normalize_text(text: str) -> str:
    """Normalize OCR text for keyword matching.

    Lowercases, collapses whitespace, removes common OCR noise.
    """
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    # Remove common OCR artifacts
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


class DocTypeDetector:
    """Detect Vietnamese document type from raw OCR text.

    Uses keyword-based scoring. Supports all doc types in KEYWORD_RULES.
    New doc types can be registered at runtime via ``register()``.

    Args:
        threshold: Minimum confidence to consider a detection valid.
            Defaults to 0.40. Lower = more permissive, Higher = more strict.
        rules: Optional custom rules dict. Defaults to built-in KEYWORD_RULES.

    Example::

        >>> from vietnerm.detector import DocTypeDetector
        >>> detector = DocTypeDetector()
        >>> result = detector.detect("CĂN CƯỚC CÔNG DÂN\\nSố: 079203030140")
        >>> print(result.doc_type)
        'cccd'
        >>> print(result.confidence)
        0.92
    """

    def __init__(
        self,
        threshold: float = _DEFAULT_THRESHOLD,
        rules: Optional[Dict[str, DocTypeRule]] = None,
    ) -> None:
        self.threshold = threshold
        self.rules: Dict[str, DocTypeRule] = dict(rules or KEYWORD_RULES)

    def register(self, doc_type: str, rule: DocTypeRule) -> None:
        """Register a new document type rule at runtime.

        Args:
            doc_type: Document type identifier (e.g., 'hop_dong').
            rule: DocTypeRule with keywords for this doc type.
        """
        self.rules[doc_type] = rule

    def detect(self, text: str) -> DetectionResult:
        """Detect document type from raw OCR text.

        Args:
            text: Raw OCR text from a Vietnamese document.

        Returns:
            DetectionResult with doc_type, confidence, and per-type scores.
        """
        text_norm = _normalize_text(text)
        raw_scores: Dict[str, float] = {}

        for doc_type, rule in self.rules.items():
            raw_scores[doc_type] = _score_text(text_norm, rule)

        total = sum(raw_scores.values())

        if total == 0:
            return DetectionResult(
                doc_type=None,
                confidence=0.0,
                scores={k: 0.0 for k in raw_scores},
                is_confident=False,
            )

        # Normalize to confidence in [0, 1]
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

        Useful for ambiguous documents or debugging.

        Args:
            text: Raw OCR text.
            n: Number of top candidates to return.

        Returns:
            List of (doc_type, confidence) tuples, sorted by confidence desc.
        """
        result = self.detect(text)
        sorted_scores = sorted(
            result.scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_scores[:n]
