"""
VietNerm NER — Zero-code auto-discovery document entity extraction.

The only class you need is ``VietNerm``.  It auto-discovers models from
HuggingFace Hub, auto-derives schema mappings from the model's ``id2label``
config, and auto-detects document types from OCR text.

**Zero-code workflow**:
  1. Add entry to ``registry/documents.yaml``
  2. Create ``templates/{doc_type}/schema.yaml`` + Jinja templates
  3. Train & push model to HuggingFace Hub (CI/CD does this automatically)
  4. SDK auto-discovers the new model — no code changes needed.

Usage::

    # Explicit doc type
    >>> from vietnerm import VietNerm
    >>> ner = VietNerm("cccd")
    >>> result = ner.extract("Ho va ten: Nguyen Van A")

    # Auto-detect doc type
    >>> ner = VietNerm()
    >>> result = ner.extract_auto(ocr_text)

    # List all available models
    >>> VietNerm.available_models()

    # Dynamic accessor (backward compat)
    >>> ner = VietNerm.for_doc("giay_khai_sinh")
"""

from typing import Any, Dict, List, Optional, Union

from ._inference.pipeline import NERPipeline
from ._inference.schema_mapper import SchemaMapper
from .detector import DocTypeDetector, DetectionResult

# Default HuggingFace username that hosts the trained models
_DEFAULT_HF_USERNAME = "ngocthanhdoan"


class VietNerm:
    """Vietnamese document NER extractor with full auto-discovery.

    This is the **only** class needed.  No shortcut subclasses required.
    Models, schema mappings, and document types are all discovered
    automatically at runtime.

    **How auto-discovery works**:

    1. ``doc_type`` → resolves to HF Hub repo ``{hf_username}/phobert-{doc_type}-ner``
    2. Model's ``config.json`` contains ``id2label`` with BIO labels
    3. ``SchemaMapper`` derives entity→field mapping from those labels
    4. Extraction returns a dict of ``{field_name: value}``

    Args:
        doc_type: Document type (e.g., 'cccd', 'giay_khai_sinh').
            If None, use ``extract_auto()`` for auto-detection or pass
            ``doc_type`` to each ``extract()`` call.
        model_path: HuggingFace Hub repo ID or local path override.
        hf_username: HuggingFace username/org hosting models.
        device: Inference device ('cpu', 'cuda', 'auto').
        max_length: Max token length. Capped at 256 (PhoBERT limit).

    Examples::

        # Simple usage — just specify doc type
        >>> ner = VietNerm("cccd")
        >>> fields = ner.extract("So: 079203030140\\nHo va ten: NGUYEN VAN A")
        >>> print(fields["id_number"])
        '079203030140'

        # Multi-doc usage — one instance, many doc types
        >>> ner = VietNerm()
        >>> cccd = ner.extract(cccd_text, doc_type="cccd")
        >>> gks  = ner.extract(gks_text,  doc_type="giay_khai_sinh")

        # Full auto — detect + extract
        >>> result = VietNerm().extract_auto(unknown_text)
        >>> print(result["doc_type"], result["fields"])
    """

    def __init__(
        self,
        doc_type: Optional[str] = None,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
        max_length: int = 256,
    ) -> None:
        self.doc_type = doc_type
        self.hf_username = hf_username
        self.device = device
        self.max_length = min(max_length, 256)
        self._pipelines: Dict[str, NERPipeline] = {}
        self._mappers: Dict[str, SchemaMapper] = {}
        self._detector: Optional[DocTypeDetector] = None

        # Eagerly load if doc_type specified
        if doc_type:
            path = model_path or self._resolve_model_path(doc_type)
            self._load_pipeline(doc_type, path)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def for_doc(
        cls,
        doc_type: str,
        model_path: Optional[str] = None,
        hf_username: str = _DEFAULT_HF_USERNAME,
        device: str = "auto",
    ) -> "VietNerm":
        """Create an extractor for a specific document type.

        This is the recommended factory method — replaces the old shortcut
        classes like ``CCCDNer``, ``GiayKhaiSinhNer``, etc.

        Args:
            doc_type: Document type identifier.
            model_path: Optional model path override.
            hf_username: HuggingFace username/org.
            device: Inference device.

        Returns:
            Configured VietNerm instance.

        Example::

            >>> ner = VietNerm.for_doc("giay_khai_sinh")
            >>> result = ner.extract(text)
        """
        return cls(
            doc_type=doc_type,
            model_path=model_path,
            hf_username=hf_username,
            device=device,
        )

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        doc_type: Optional[Union[str, DetectionResult]] = None,
        validate: bool = True,
    ) -> Dict[str, str]:
        """Extract structured entities from document text.

        Args:
            text: Raw document text (e.g., OCR output).
            doc_type: Document type. Uses instance default if None.
            validate: Whether to validate extracted entities.

        Returns:
            Dict mapping field names to extracted values.
            Fields with no match are empty strings.

        Raises:
            ValueError: If no doc_type specified.
        """
        dt = self._resolve_doc_type_input(doc_type) or self.doc_type
        if not dt:
            raise ValueError(
                "doc_type must be specified either in constructor or extract() call. "
                "Use extract_auto() for automatic detection."
            )
        self._ensure_loaded(dt)
        raw = self._pipelines[dt].predict(text)
        return self._mappers[dt].map_entities(raw, validate=validate, clean=True)

    def extract_with_confidence(
        self,
        text: str,
        doc_type: Optional[Union[str, DetectionResult]] = None,
        validate: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract entities with confidence scores.

        Args:
            text: Raw document text.
            doc_type: Document type.
            validate: Whether to validate.

        Returns:
            Dict mapping field names to ``{"value": str, "confidence": float}``.
        """
        dt = self._resolve_doc_type_input(doc_type) or self.doc_type
        if not dt:
            raise ValueError("doc_type required")
        self._ensure_loaded(dt)
        raw = self._pipelines[dt].predict(text)
        return self._mappers[dt].map_entities_with_confidence(
            raw, validate=validate, clean=True
        )

    def extract_raw(
        self,
        text: str,
        doc_type: Optional[Union[str, DetectionResult]] = None,
    ) -> List[Dict]:
        """Get raw NER predictions without schema mapping.

        Args:
            text: Raw document text.
            doc_type: Document type.

        Returns:
            List of entity dicts with type, text, start, end, confidence.
        """
        dt = self._resolve_doc_type_input(doc_type) or self.doc_type
        if not dt:
            raise ValueError("doc_type required")
        self._ensure_loaded(dt)
        return self._pipelines[dt].predict(text)

    def extract_auto(
        self,
        text: str,
        validate: bool = True,
        detection_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Auto-detect document type then extract entities.

        Combines detection + extraction in one call.

        Args:
            text: Raw OCR text.
            validate: Whether to validate entities.
            detection_threshold: Minimum confidence for detection.

        Returns:
            Dict with keys: ``doc_type``, ``detection_confidence``,
            ``detection_scores``, ``fields``.
        """
        detection = self.detect_doc_type(text, threshold=detection_threshold)

        result: Dict[str, Any] = {
            "doc_type": detection.doc_type,
            "detection_confidence": detection.confidence,
            "detection_scores": detection.scores,
            "fields": {},
        }

        if detection.is_confident and detection.doc_type:
            dt = detection.doc_type
            self._ensure_loaded(dt)
            raw = self._pipelines[dt].predict(text)
            result["fields"] = self._mappers[dt].map_entities(
                raw, validate=validate, clean=True
            )

        return result

    def detect_doc_type(
        self,
        text: str,
        threshold: float = 0.25,
    ) -> DetectionResult:
        """Detect document type from raw OCR text.

        Uses TF-IDF + cosine similarity — fast, no model loading.

        Args:
            text: Raw OCR text.
            threshold: Minimum confidence threshold.

        Returns:
            DetectionResult with doc_type, confidence, scores.
        """
        if self._detector is None:
            self._detector = DocTypeDetector(threshold=threshold)
        else:
            self._detector.threshold = threshold
        return self._detector.detect(text)

    # ------------------------------------------------------------------
    # Discovery methods
    # ------------------------------------------------------------------

    @classmethod
    def available_models(
        cls,
        hf_username: str = _DEFAULT_HF_USERNAME,
    ) -> List[Dict[str, str]]:
        """List all available NER models on HuggingFace Hub.

        Discovers models matching ``{hf_username}/phobert-*-ner``.

        Args:
            hf_username: HuggingFace username/org.

        Returns:
            List of dicts with 'doc_type', 'repo_id', 'name'.

        Example::

            >>> for m in VietNerm.available_models():
            ...     print(f"{m['doc_type']:25s} {m['repo_id']}")
            cccd                      ngocthanhdoan/phobert-cccd-ner
            giay_khai_sinh            ngocthanhdoan/phobert-giay_khai_sinh-ner
            ...
        """
        try:
            from huggingface_hub import list_models
            results = []
            for model in list_models(author=hf_username, search="phobert-ner"):
                repo_id = model.id if hasattr(model, "id") else str(model)
                name_part = repo_id.split("/")[-1]
                if name_part.startswith("phobert-") and name_part.endswith("-ner"):
                    doc_type = name_part[len("phobert-"):-len("-ner")]
                    results.append({
                        "doc_type": doc_type,
                        "repo_id": repo_id,
                        "name": doc_type.replace("_", " ").title(),
                    })
            return sorted(results, key=lambda x: x["doc_type"])
        except Exception:
            return []

    @classmethod
    def available_doc_types(
        cls,
        hf_username: str = _DEFAULT_HF_USERNAME,
    ) -> List[str]:
        """List available document type identifiers.

        Returns:
            Sorted list of doc_type strings.

        Example::

            >>> VietNerm.available_doc_types()
            ['cccd', 'giay_khai_sinh', 'giay_ra_vien', 'gplx', 'vehicle_registration']
        """
        return [m["doc_type"] for m in cls.available_models(hf_username)]

    def get_schema(
        self, doc_type: Optional[Union[str, DetectionResult]] = None
    ) -> Dict[str, str]:
        """Get the field schema for a document type.

        Loads the model if needed and returns the auto-discovered mapping.

        Args:
            doc_type: Document type. Uses instance default if None.

        Returns:
            Dict mapping entity types to field names.

        Example::

            >>> VietNerm("cccd").get_schema()
            {'ID_NUMBER': 'id_number', 'FULL_NAME': 'full_name', ...}
        """
        dt = self._resolve_doc_type_input(doc_type) or self.doc_type
        if not dt:
            raise ValueError("doc_type required")
        self._ensure_loaded(dt)
        return dict(self._mappers[dt].entity_to_field)

    def get_fields(
        self, doc_type: Optional[Union[str, DetectionResult]] = None
    ) -> List[str]:
        """Get the list of extractable fields for a document type.

        Args:
            doc_type: Document type.

        Returns:
            Sorted list of field names.

        Example::

            >>> VietNerm("cccd").get_fields()
            ['date_of_birth', 'date_of_expiry', 'full_name', ...]
        """
        dt = self._resolve_doc_type_input(doc_type) or self.doc_type
        if not dt:
            raise ValueError("doc_type required")
        self._ensure_loaded(dt)
        return sorted(self._mappers[dt].expected_fields)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self, doc_type: str) -> str:
        """Resolve doc_type to HF Hub repo ID."""
        return f"{self.hf_username}/phobert-{doc_type}-ner"

    @staticmethod
    def _resolve_doc_type_input(
        doc_type: Optional[Union[str, DetectionResult]]
    ) -> Optional[str]:
        """Normalize doc_type input to a string identifier.

        Accepts either a plain string or a ``DetectionResult`` object.
        """
        if isinstance(doc_type, DetectionResult):
            return doc_type.doc_type
        if isinstance(doc_type, str):
            return doc_type
        return None

    def _ensure_loaded(self, doc_type: str) -> None:
        """Ensure pipeline + mapper are loaded for doc_type."""
        if doc_type not in self._pipelines:
            path = self._resolve_model_path(doc_type)
            self._load_pipeline(doc_type, path)

    def _load_pipeline(self, doc_type: str, model_path: str) -> None:
        """Load NER pipeline and auto-configure schema mapper."""
        if doc_type in self._pipelines:
            return

        pipeline = NERPipeline(
            model_path=model_path,
            device=self.device,
            max_length=self.max_length,
        )
        self._pipelines[doc_type] = pipeline

        # Auto-derive mapping from model's id2label — zero config needed
        self._mappers[doc_type] = SchemaMapper(
            doc_type=doc_type,
            id2label=pipeline.id2label,
            model_repo_id=model_path if "/" in model_path else None,
        )

    # ------------------------------------------------------------------
    # Backward compatibility aliases
    # ------------------------------------------------------------------

    # Old method name kept as alias
    list_available_models = available_models

    def __repr__(self) -> str:
        loaded = list(self._pipelines.keys())
        return (
            f"VietNerm(doc_type={self.doc_type!r}, "
            f"loaded={loaded}, hf={self.hf_username!r})"
        )
