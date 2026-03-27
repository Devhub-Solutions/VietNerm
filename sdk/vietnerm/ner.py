"""
VietNerm NER classes for Vietnamese document entity extraction.

Provides high-level API wrapping the inference pipeline:
  - VietNerm: Generic extractor, auto-detect or specify document type
  - CCCDNer: Shortcut for CCCD (Citizen ID Card) extraction
  - GiayRaVienNer: Shortcut for Giấy ra viện (Hospital Discharge) extraction
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path for inference module access
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from inference.pipeline import NERPipeline
from inference.schema_mapper import SchemaMapper


class VietNerm:
    """Vietnamese document NER extractor.

    Wraps the inference pipeline and schema mapper for easy entity extraction
    from Vietnamese administrative and medical documents.

    Args:
        doc_type: Document type ('cccd', 'giay_ra_vien', etc.).
            If None, must be specified per call.
        model_path: Path to trained model or HuggingFace hub ID.
            Defaults to models/phobert/{doc_type}/.
        device: Device for inference ('cpu', 'cuda', 'auto').
        max_length: Maximum token sequence length.

    Example:
        >>> ner = VietNerm(doc_type="cccd")
        >>> result = ner.extract("Họ và tên: Nguyễn Văn A\\nNgày sinh: 01/01/1990")
        >>> print(result)
        {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}
    """

    # Default HuggingFace Hub model IDs per document type
    DEFAULT_HUB_MODELS: Dict[str, str] = {
        "cccd": "vietnerm/phobert-cccd-ner",
        "giay_ra_vien": "vietnerm/phobert-giay-ra-vien-ner",
    }

    def __init__(
        self,
        doc_type: Optional[str] = None,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.doc_type = doc_type
        self.device = device
        self.max_length = max_length
        self._pipelines: Dict[str, NERPipeline] = {}
        self._mappers: Dict[str, SchemaMapper] = {}

        if doc_type and model_path:
            self._load_pipeline(doc_type, model_path)
        elif doc_type:
            resolved = self._resolve_model_path(doc_type)
            self._load_pipeline(doc_type, resolved)

    def _resolve_model_path(self, doc_type: str) -> str:
        """Resolve model path: try local first, then HuggingFace Hub.

        Args:
            doc_type: Document type identifier.

        Returns:
            Model path string (local or hub ID).
        """
        # Try local model directory
        local_path = _project_root / "models" / "phobert" / doc_type
        if local_path.exists() and (local_path / "config.json").exists():
            return str(local_path)

        # Fall back to HuggingFace Hub
        if doc_type in self.DEFAULT_HUB_MODELS:
            return self.DEFAULT_HUB_MODELS[doc_type]

        raise FileNotFoundError(
            f"No model found for doc_type='{doc_type}'. "
            f"Expected local model at: {local_path} "
            f"or set model_path explicitly."
        )

    def _load_pipeline(self, doc_type: str, model_path: str) -> None:
        """Load pipeline and mapper for a document type.

        Args:
            doc_type: Document type identifier.
            model_path: Model path or HuggingFace hub ID.
        """
        if doc_type not in self._pipelines:
            self._pipelines[doc_type] = NERPipeline(
                model_path=model_path,
                device=self.device,
                max_length=self.max_length,
            )
            self._mappers[doc_type] = SchemaMapper(doc_type=doc_type)

    def extract(
        self,
        text: str,
        doc_type: Optional[str] = None,
        validate: bool = True,
    ) -> Dict[str, str]:
        """Extract structured entities from document text.

        Args:
            text: Raw document text (e.g., OCR output).
            doc_type: Document type. Uses instance doc_type if None.
            validate: Whether to validate extracted entities.

        Returns:
            Dict mapping field names to extracted values.

        Raises:
            ValueError: If no doc_type is specified.
        """
        dt = doc_type or self.doc_type
        if not dt:
            raise ValueError(
                "doc_type must be specified either in constructor or extract() call"
            )

        if dt not in self._pipelines:
            resolved = self._resolve_model_path(dt)
            self._load_pipeline(dt, resolved)

        raw_entities = self._pipelines[dt].predict(text)
        return self._mappers[dt].map_entities(
            raw_entities, validate=validate, clean=True
        )

    def extract_with_confidence(
        self,
        text: str,
        doc_type: Optional[str] = None,
        validate: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract entities with confidence scores.

        Args:
            text: Raw document text.
            doc_type: Document type.
            validate: Whether to validate extracted entities.

        Returns:
            Dict mapping field names to {"value": str, "confidence": float}.
        """
        dt = doc_type or self.doc_type
        if not dt:
            raise ValueError(
                "doc_type must be specified either in constructor or extract() call"
            )

        if dt not in self._pipelines:
            resolved = self._resolve_model_path(dt)
            self._load_pipeline(dt, resolved)

        raw_entities = self._pipelines[dt].predict(text)
        return self._mappers[dt].map_entities_with_confidence(
            raw_entities, validate=validate, clean=True
        )

    def extract_raw(
        self,
        text: str,
        doc_type: Optional[str] = None,
    ) -> List[Dict]:
        """Get raw NER predictions without schema mapping.

        Args:
            text: Raw document text.
            doc_type: Document type.

        Returns:
            List of raw entity dicts with type, text, start, end, confidence.
        """
        dt = doc_type or self.doc_type
        if not dt:
            raise ValueError(
                "doc_type must be specified either in constructor or extract() call"
            )

        if dt not in self._pipelines:
            resolved = self._resolve_model_path(dt)
            self._load_pipeline(dt, resolved)

        return self._pipelines[dt].predict(text)


class CCCDNer(VietNerm):
    """Shortcut NER extractor for CCCD (Căn cước công dân) documents.

    Args:
        model_path: Optional model path override.
        device: Device for inference.

    Example:
        >>> ner = CCCDNer()
        >>> result = ner.extract("Số: 079203030140\\nHọ và tên: NGUYỄN VĂN A")
        >>> print(result["name"])
        "Nguyễn Văn A"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(
            doc_type="cccd",
            model_path=model_path,
            device=device,
            max_length=256,
        )


class GiayRaVienNer(VietNerm):
    """Shortcut NER extractor for Giấy ra viện (Hospital Discharge) documents.

    Args:
        model_path: Optional model path override.
        device: Device for inference.

    Example:
        >>> ner = GiayRaVienNer()
        >>> result = ner.extract("Họ tên người bệnh: Lê Thị Hằng\\nChẩn đoán: Viêm phổi")
        >>> print(result["patient_name"])
        "Lê Thị Hằng"
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
    ) -> None:
        super().__init__(
            doc_type="giay_ra_vien",
            model_path=model_path,
            device=device,
            max_length=512,
        )
