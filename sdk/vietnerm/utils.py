"""
Utility functions for VietNerm SDK.

Text preprocessing, normalization, and helper functions for Vietnamese NER.
"""

import re
import unicodedata
from typing import List


def normalize_unicode(text: str) -> str:
    """Normalize Vietnamese Unicode text to NFC form.

    Vietnamese text can have composed or decomposed diacritics.
    NFC normalization ensures consistent representation.

    Args:
        text: Input text to normalize.

    Returns:
        NFC-normalized text.
    """
    return unicodedata.normalize("NFC", text)


def clean_ocr_text(text: str) -> str:
    """Clean common OCR artifacts from document text.

    Args:
        text: Raw OCR text.

    Returns:
        Cleaned text.
    """
    # Normalize unicode
    text = normalize_unicode(text)

    # Normalize whitespace (but preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Remove zero-width characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # Clean up multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def segment_sentences(text: str) -> List[str]:
    """Split Vietnamese text into sentences/lines.

    Splits on newlines and common Vietnamese sentence boundaries.

    Args:
        text: Input text.

    Returns:
        List of sentences/lines.
    """
    lines = text.split("\n")
    return [line.strip() for line in lines if line.strip()]


def normalize_name(name: str) -> str:
    """Normalize a Vietnamese person name.

    Applies title casing appropriate for Vietnamese names.

    Args:
        name: Raw name text.

    Returns:
        Normalized name.
    """
    # Handle all-caps or all-lower
    name = name.strip()
    if name == name.upper() or name == name.lower():
        return name.title()
    return name


def normalize_date(date_str: str) -> str:
    """Normalize Vietnamese date format to dd/mm/yyyy.

    Handles various input formats commonly seen in Vietnamese documents.

    Args:
        date_str: Raw date string.

    Returns:
        Normalized date string in dd/mm/yyyy format, or original if unparseable.
    """
    date_str = date_str.strip()

    # Already in dd/mm/yyyy
    if re.match(r"^\d{2}/\d{2}/\d{4}$", date_str):
        return date_str

    # Try "ngày X tháng Y năm Z" format
    match = re.search(
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
        date_str,
        re.IGNORECASE,
    )
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    # Try "X/Y/Z" with varying digits
    match = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
    if match:
        day, month, year = match.groups()
        return f"{int(day):02d}/{int(month):02d}/{year}"

    return date_str


def is_vietnamese_text(text: str) -> bool:
    """Check if text contains Vietnamese characters.

    Args:
        text: Input text.

    Returns:
        True if text contains Vietnamese diacritical characters.
    """
    vietnamese_pattern = re.compile(
        r"[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
        r"ùúủũụưừứửữựỳýỷỹỵ"
        r"ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
        r"ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]"
    )
    return bool(vietnamese_pattern.search(text))
