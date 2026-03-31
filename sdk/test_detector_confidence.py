"""Tests for detector confidence behavior on OCR-noisy text."""

from vietnerm.detector import DocTypeDetector, DocTypeRule


def test_detector_accepts_strong_keyword_with_soft_threshold():
    rules = {
        "cccd": DocTypeRule(
            strong_keywords=["căn cước công dân"],
            weak_keywords=["quốc tịch", "giới tính", "nơi thường trú"],
        ),
        "gplx": DocTypeRule(
            strong_keywords=["giấy phép lái xe"],
            weak_keywords=["hạng", "ngày cấp", "có giá trị đến"],
        ),
    }

    detector = DocTypeDetector(rules=rules, index=None, threshold=0.25)
    text = (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n"
        "CĂN CƯỚC CÔNG DÂN\n"
        "Quốc tịch: Việt Nam\n"
        "Có giả trị đến: 05/04/2029\n"
    )

    result = detector.detect(text)
    assert result.doc_type == "cccd"
    assert result.is_confident is True


def test_detector_can_still_return_none_without_strong_hit():
    rules = {
        "cccd": DocTypeRule(strong_keywords=["căn cước công dân"], weak_keywords=["quốc tịch"]),
        "gplx": DocTypeRule(strong_keywords=["giấy phép lái xe"], weak_keywords=["hạng"]),
    }
    detector = DocTypeDetector(rules=rules, index=None, threshold=0.25)
    result = detector.detect("dữ liệu OCR rất mờ và không có từ khóa đặc trưng")
    assert result.doc_type is None
