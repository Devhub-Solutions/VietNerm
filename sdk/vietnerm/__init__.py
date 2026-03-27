"""
VietNerm SDK - Vietnamese Document NER Extraction.

Extract structured entities from Vietnamese documents using PhoBERT-based NER models.

Usage:
    >>> from vietnerm import VietNerm
    >>> ner = VietNerm(doc_type="cccd")
    >>> result = ner.extract("Họ và tên: Nguyễn Văn A\\nNgày sinh: 01/01/1990")
    >>> print(result)
    {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}
"""

from .ner import VietNerm, CCCDNer, GiayRaVienNer

__version__ = "0.1.0"

__all__ = [
    "VietNerm",
    "CCCDNer",
    "GiayRaVienNer",
    "__version__",
]
