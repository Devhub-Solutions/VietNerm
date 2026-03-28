"""
VietNerm SDK - Vietnamese Document NER Extraction.

Extract structured entities from Vietnamese administrative and medical documents
using PhoBERT-based NER models hosted on HuggingFace Hub.

Quick start::

    >>> from vietnerm import CCCDNer
    >>> ner = CCCDNer()
    >>> result = ner.extract("Họ và tên: Nguyễn Văn A\\nNgày sinh: 01/01/1990")
    >>> print(result)
    {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}

Using generic extractor::

    >>> from vietnerm import VietNerm
    >>> ner = VietNerm(doc_type="giay_ra_vien")
    >>> result = ner.extract("Họ tên người bệnh: Lê Thị Hằng")

Using custom HuggingFace model::

    >>> ner = VietNerm(doc_type="cccd", model_path="my-org/my-cccd-model")
    >>> result = ner.extract(text)
"""

from .ner import VietNerm, CCCDNer, GiayRaVienNer, VehicleRegistrationNer
from .detector import DocTypeDetector, DetectionResult, DocTypeRule

__version__ = "0.1.0"
__author__ = "VietNerm Team"

__all__ = [
    "VietNerm",
    "CCCDNer",
    "GiayRaVienNer",
    "VehicleRegistrationNer",
    "DocTypeDetector",
    "DetectionResult",
    "DocTypeRule",
    "__version__",
]
