"""
VietNerm NER classes for Vietnamese document entity extraction.

Provides high-level API wrapping the inference pipeline:
  - VietNerm: Generic extractor, auto-detect or specify document type
  - CCCDNer: Shortcut for CCCD (Citizen ID Card) extraction
  - GiayRaVienNer: Shortcut for Giấy ra viện (Hospital Discharge) extraction
  - VehicleRegistrationNer: Shortcut for Đăng ký xe extraction

Models are loaded from HuggingFace Hub by default:
  https://huggingface.co/{hf_username}/phobert-{doc_type}-ner
"""

from typing import Any, Dict, List, Optional

from ._inference.pipeline import NERPipeline
from ._inference.schema_mapper import SchemaMapper

# Default HuggingFace username that hosts the trained models
_DEFAULT_HF_USERNAME = "ngocthanhdoan"


class VietNerm:
    """Vietnamese document NER extractor.

    Wraps the inference pipeline and schema mapper for easy entity extraction
    from Vietnamese administrative and medical documents.

    Models are loaded from HuggingFace Hub by default. You can override
    with a local path via ``model_path``.

    Args:
        doc_type: Document type ('cccd', 'giay_ra_vien', 'vehicle_registration', etc.).
            If None, must be specified per call.
        model_path: HuggingFace Hub repo ID or local path to the trained model.
            If None, resolved automatically as ``{hf_username}/phobert-{doc_type}-ner``.
        hf_username: HuggingFace username/org that hosts the models.
            Defaults to 'ngocthanhdoan'.
        device: Device for inference ('cpu', 'cuda', 'auto').
        max_length: Maximum token sequence length.

    Example::

        >>> from vietnerm import VietNerm
        >>> ner = VietNerm(doc_type="cccd")
        >>> result = ner.extract("Họ và tên: Nguyễn Văn A\\nNgày sinh: 01/01/1990")
        >>> print(result)
        {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}
    """

    def __init__(
        self,
        doc_type: Optional[str] = None,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        self.doc_type = doc_type
        self.hf_username = hf_username
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
        """Resolve model to HuggingFace Hub ID.

        Naming convention: ``{hf_username}/phobert-{doc_type}-ner``
        (dashes replace underscores in doc_type).

        Args:
            doc_type: Document type identifier.

        Returns:
            HuggingFace Hub repo ID string.
        """
        # Naming convention on HuggingFace Hub: phobert-{doc_type}-ner
        # doc_type keeps underscores as-is (e.g. giay_ra_vien, vehicle_registration)
        return f"{self.hf_username}/phobert-{doc_type}-ner"

    def _load_pipeline(self, doc_type: str, model_path: str) -> None:
        """Load pipeline and mapper for a document type.

        Args:
            doc_type: Document type identifier.
            model_path: HuggingFace Hub ID or local model path.
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

    Loads model from: ``{hf_username}/phobert-cccd-ner``

    Args:
        model_path: Optional model path or Hub ID override.
        hf_username: HuggingFace username/org.
        device: Device for inference.

    Example::

        >>> from vietnerm import CCCDNer
        >>> ner = CCCDNer()
        >>> result = ner.extract("Số: 079203030140\\nHọ và tên: NGUYỄN VĂN A")
        >>> print(result["name"])
        'Nguyễn Văn A'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
    ) -> None:
        super().__init__(
            doc_type="cccd",
            model_path=model_path,
            hf_username=hf_username,
            device=device,
            max_length=256,
        )


class GiayRaVienNer(VietNerm):
    """Shortcut NER extractor for Giấy ra viện (Hospital Discharge) documents.

    Loads model from: ``{hf_username}/phobert-giay-ra-vien-ner``

    Args:
        model_path: Optional model path or Hub ID override.
        hf_username: HuggingFace username/org.
        device: Device for inference.

    Example::

        >>> from vietnerm import GiayRaVienNer
        >>> ner = GiayRaVienNer()
        >>> result = ner.extract("Họ tên người bệnh: Lê Thị Hằng\\nChẩn đoán: Viêm phổi")
        >>> print(result["patient_name"])
        'Lê Thị Hằng'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
    ) -> None:
        super().__init__(
            doc_type="giay_ra_vien",
            model_path=model_path,
            hf_username=hf_username,
            device=device,
            max_length=512,
        )


class VehicleRegistrationNer(VietNerm):
    """Shortcut NER extractor for Đăng ký xe (Vehicle Registration) documents.

    Loads model from: ``{hf_username}/phobert-vehicle-registration-ner``

    Args:
        model_path: Optional model path or Hub ID override.
        hf_username: HuggingFace username/org.
        device: Device for inference.

    Example::

        >>> from vietnerm import VehicleRegistrationNer
        >>> ner = VehicleRegistrationNer()
        >>> result = ner.extract("Biển số: 51A-123.45\\nChủ xe: Nguyễn Văn A")
        >>> print(result["plate_number"])
        '51A-123.45'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
    ) -> None:
        super().__init__(
            doc_type="vehicle_registration",
            model_path=model_path,
            hf_username=hf_username,
            device=device,
            max_length=512,
        )
