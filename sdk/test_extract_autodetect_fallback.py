"""Tests for auto-detect fallback in extract APIs."""

from vietnerm import DetectionResult, VietNerm


class _DummyPipeline:
    def predict(self, text: str):
        return [{"type": "ID_NUMBER", "text": "024304001506", "confidence": 0.99}]


class _DummyMapper:
    def map_entities(self, raw, validate=True, clean=True):
        return {"id_number": raw[0]["text"]}

    def map_entities_with_confidence(self, raw, validate=True, clean=True):
        return {"id_number": {"value": raw[0]["text"], "confidence": raw[0]["confidence"]}}


def test_extract_auto_detects_when_doc_type_not_provided(monkeypatch):
    ner = VietNerm()
    ner._pipelines["cccd"] = _DummyPipeline()
    ner._mappers["cccd"] = _DummyMapper()

    monkeypatch.setattr(
        ner,
        "detect_doc_type",
        lambda text, threshold=0.25: DetectionResult(
            doc_type="cccd",
            confidence=0.91,
            scores={"cccd": 0.91},
            is_confident=True,
            method="keyword",
        ),
    )

    result = ner.extract("dummy text")
    assert result["id_number"] == "024304001506"


def test_extract_raises_helpful_error_when_auto_detect_not_confident(monkeypatch):
    ner = VietNerm()
    monkeypatch.setattr(
        ner,
        "detect_doc_type",
        lambda text, threshold=0.25: DetectionResult(
            doc_type=None,
            confidence=0.12,
            scores={"cccd": 0.12, "gplx": 0.11},
            is_confident=False,
            method="keyword",
        ),
    )

    try:
        ner.extract("dummy text")
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        msg = str(exc)
        assert "Auto-detection is currently not confident enough" in msg
        assert "Best scores" in msg
