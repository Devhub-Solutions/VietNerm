"""
VietNerm SDK — Zero-code Vietnamese Document NER Extraction.

Extract structured entities from Vietnamese administrative and medical documents
using PhoBERT-based NER models hosted on HuggingFace Hub.

**Zero-code auto-discovery**: Just upload template + schema + registry entry,
train the model, and the SDK auto-discovers everything.  No code changes needed.

Quick start::

    >>> from vietnerm import VietNerm
    >>> ner = VietNerm("cccd")
    >>> result = ner.extract("Họ và tên: Nguyễn Văn A\\nNgày sinh: 01/01/1990")
    >>> print(result)
    {'id_number': '', 'full_name': 'Nguyễn Văn A', 'date_of_birth': '01/01/1990', ...}

Auto-detect document type::

    >>> ner = VietNerm()
    >>> result = ner.extract_auto(ocr_text)
    >>> print(result["doc_type"], result["fields"])

List available models::

    >>> VietNerm.available_models()
    [{'doc_type': 'cccd', 'repo_id': 'ngocthanhdoan/phobert-cccd-ner', ...}, ...]

Factory method::

    >>> ner = VietNerm.for_doc("giay_khai_sinh")
    >>> result = ner.extract(text)
"""

from .ner import VietNerm
from .detector import DocTypeDetector, DetectionResult, DocTypeRule
from .registry import ModelRegistry
from .download import DownloadConfig
__version__ = "0.2.6"
__author__ = "VietNerm Team"

__all__ = [
    "VietNerm",
    "DocTypeDetector",
    "DetectionResult",
    "DocTypeRule",
    "ModelRegistry",
    "DownloadConfig",
    "__version__",
]
