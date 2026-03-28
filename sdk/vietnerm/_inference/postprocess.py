"""
Post-processing utilities for NER predictions.

Handles:
  - Merging sub-token predictions (PhoBERT BPE)
  - B-/I- tag merging
  - Entity boundary cleanup
  - Confidence scoring
"""

import re
from typing import Dict, List, Optional, Tuple

# Gender normalization map (handles common OCR errors)
GENDER_NORMALIZE: Dict[str, str] = {
    "NỮ": "Nữ", "NU": "Nữ", "NO": "Nữ",
    "N0": "Nữ", "NƯ": "Nữ", "NỪ": "Nữ",
    "NAM": "Nam",
}


def _normalize_etype(tag_name: str) -> str:
    """Normalize entity type from tag name.

    Supports two label schemes:
      - New (HuggingFace Hub): 'full_name', 'date_of_birth'  -> 'FULL_NAME'
      - Old (local training):  'full_name_VALUE'             -> 'FULL_NAME'
    """
    return tag_name.replace("_VALUE", "").upper()


def merge_subtoken_predictions(
    tokens: List[str],
    labels: List[str],
    confidences: List[float],
    tokens_with_pos: List[Tuple[str, int, int]],
) -> List[Dict]:
    """Merge B-/I- tagged tokens into entity spans.

    Supports both label schemes:
      - New (HuggingFace Hub): B-full_name, I-full_name
      - Old (local training):  B-full_name_VALUE, I-full_name_VALUE

    Args:
        tokens: List of whitespace tokens.
        labels: BIO labels for each token.
        confidences: Confidence score for each token prediction.
        tokens_with_pos: Tokens with (token, start, end) char offsets.

    Returns:
        List of entity dicts with type, text, start, end, confidence.
    """
    entities: List[Dict] = []
    current_ent: Optional[Dict] = None

    for i, (tok, label) in enumerate(zip(tokens, labels)):
        _, tok_start, tok_end = tokens_with_pos[i]
        conf = confidences[i] if i < len(confidences) else 0.0

        # Skip O labels
        if label == "O":
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        parts = label.split("-", 1)
        if len(parts) != 2:
            if current_ent:
                entities.append(current_ent)
                current_ent = None
            continue

        prefix, tag_name = parts
        # Normalize entity type — supports both old (_VALUE) and new schemes
        etype = _normalize_etype(tag_name)

        if prefix == "B":
            if current_ent:
                entities.append(current_ent)
            current_ent = {
                "type": etype,
                "text": tok,
                "start": tok_start,
                "end": tok_end,
                "label": label,
                "confidence": conf,
                "_conf_sum": conf,
                "_conf_count": 1,
            }
        elif prefix == "I":
            if current_ent and current_ent["type"] == etype:
                current_ent["text"] += " " + tok
                current_ent["end"] = tok_end
                current_ent["_conf_sum"] += conf
                current_ent["_conf_count"] += 1
            else:
                # I- without preceding B- → treat as B-
                if current_ent:
                    entities.append(current_ent)
                current_ent = {
                    "type": etype,
                    "text": tok,
                    "start": tok_start,
                    "end": tok_end,
                    "label": f"B-{tag_name}",
                    "confidence": conf,
                    "_conf_sum": conf,
                    "_conf_count": 1,
                }

    if current_ent:
        entities.append(current_ent)

    # Compute average confidence and clean up internal fields
    for ent in entities:
        ent["confidence"] = compute_confidence(
            ent.pop("_conf_sum"), ent.pop("_conf_count")
        )

    return entities


def compute_confidence(conf_sum: float, conf_count: int) -> float:
    """Compute average confidence from accumulated sum and count.

    Args:
        conf_sum: Sum of token-level confidence scores.
        conf_count: Number of tokens.

    Returns:
        Average confidence rounded to 4 decimal places.
    """
    if conf_count == 0:
        return 0.0
    return round(conf_sum / conf_count, 4)


def clean_entity_boundaries(entities: List[Dict]) -> List[Dict]:
    """Clean up entity text boundaries and normalize values.

    Applies:
      - Strip whitespace
      - Gender normalization
      - Name title-casing
      - Deduplication (keep first occurrence per entity type)

    Args:
        entities: Raw entity list from merge_subtoken_predictions.

    Returns:
        Cleaned and deduplicated entity list.
    """
    cleaned: List[Dict] = []
    seen_types: set = set()

    for ent in entities:
        t = ent["type"]
        text = ent["text"].strip()

        if t in seen_types:
            # Special case: Merge multiple address or notes regions
            if "ADDRESS" in t or "NOTES" in t or "DIAGNOSIS" in t:
                # Find existing entry and append
                for existing in cleaned:
                    if existing["type"] == t:
                        existing["text"] += ", " + text
                        existing["end"] = ent.get("end", 0)
                        break
            continue

        # Normalize gender (type is already uppercase)
        if "GENDER" in t:
            text = GENDER_NORMALIZE.get(text.upper(), text)

        # Title-case names (handle ALL-CAPS or mixed-case OCR errors)
        if "NAME" in t:
            # Only title-case if contains multiple lowercase (to preserve "Nguyễn Văn A" but fix "NGọc")
            # Actually, the safest is to just .title() for names generally in Vietnamese
            words = text.split()
            text = " ".join([w.capitalize() for w in words])

        ent_clean = {
            "type": t,
            "text": text,
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
            "confidence": ent.get("confidence", 0.0),
        }
        cleaned.append(ent_clean)
        seen_types.add(t)

    return cleaned


def validate_entity(entity_type: str, text: str) -> bool:
    """Validate extracted entity value based on entity type.

    Args:
        entity_type: The NER entity type.
        text: The extracted text value.

    Returns:
        True if the entity passes validation.
    """
    validators: Dict[str, callable] = {
        "PATIENT_DOB": lambda x: bool(
            re.match(r"^\d{2}/\d{2}/\d{4}$|^\d{4}$", x.strip())
        ),
        "DOB": lambda x: bool(re.match(r"^\d{2}/\d{2}/\d{4}$", x.strip())),
        "ADMISSION_DATE": lambda x: bool(re.search(r"\d{1,2}.*\d{4}", x.strip())),
        "DISCHARGE_DATE": lambda x: bool(re.search(r"\d{1,2}.*\d{4}", x.strip())),
        "DATE_OF_EXPIRY": lambda x: bool(
            re.match(r"^\d{2}/\d{2}/\d{4}$", x.strip())
        ),
        "PATIENT_GENDER": lambda x: (
            GENDER_NORMALIZE.get(x.strip().upper(), x.strip()) in ("Nam", "Nữ")
        ),
        "GENDER": lambda x: (
            GENDER_NORMALIZE.get(x.strip().upper(), x.strip()) in ("Nam", "Nữ")
        ),
        "PATIENT_ETHNICITY": lambda x: 1 <= len(x.split()) <= 3,
        "PATIENT_NAME": lambda x: 2 <= len(x.split()) <= 7 and not re.search(r"\d", x),
        "FULL_NAME": lambda x: (
            bool(
                re.match(
                    r"^[A-ZÀÁẢÃẠĂẮẶẲẴÂẤẦẨẪẬĐÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
                    r"ùúụủũưừứựửữỳýỵỷỹ\s]+$",
                    x.strip(),
                    re.IGNORECASE,
                )
            )
            and 2 <= len(x.split()) <= 6
        ),
        "ID_NUMBER": lambda x: bool(re.match(r"^\d{9}$|^\d{12}$", x.strip())),
        "BHXH_CODE": lambda x: len(re.sub(r"\s", "", x)) >= 8,
        "MEDICAL_CODE": lambda x: bool(re.search(r"\d", x)),
        "DIAGNOSIS": lambda x: len(x.strip()) >= 5,
        "TREATMENT_METHOD": lambda x: len(x.strip()) >= 5,
    }

    validator = validators.get(entity_type)
    if validator is None:
        return True
    return validator(text)


def filter_validated_entities(entities: List[Dict]) -> List[Dict]:
    """Filter entities that pass type-specific validation.

    Args:
        entities: Entity list (already cleaned).

    Returns:
        Filtered list with only valid entities.
    """
    return [
        ent for ent in entities
        if validate_entity(ent["type"], ent["text"])
    ]
