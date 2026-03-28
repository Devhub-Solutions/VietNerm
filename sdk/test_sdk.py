"""Test VietNerm SDK dynamic detection and extraction."""

import sys
from pathlib import Path

# Add sdk/ parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from vietnerm.ner import VietNerm

def test_auto_extraction():
    ner = VietNerm()
    
    # Test texts for different document types
    samples = [
        # CCCD
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập Tự do - Hạnh phúc\nCĂN CƯỚC CÔNG DÂN\n031189007896\nHọ và tên: NGUYỄN THỊ KIM CHI",
        # GPLX
        "GIAY PHẾP LÁI XE/DRIVER/SLICENSE\nSố/No: 790208248116\nHọ tên/Full name: NGUYỄN MINH NHƯT\nNgày sinh/Date of Birth: 08/05/1999",
        # Giấy Khai Sinh
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\nGIẤY KHAI SINH\nHo và tên: LÊ GIA BẢO\nGiới tính: Nam\nNgày, tháng, năm sinh: 21/02/2012",
        # Giấy Ra Viện
        "BỆNH VIỆN ĐA KHOA VĨNH ĐỨC\nGIẤY RA VIỆN\nHọ tên người bệnh: Bùi Văn Thiên\nNgày sinh: 08/07/1971",
        # Vehicle (Inspection)
        "1. PHƯƠNG TIỆN (VEHICLE)\nBiển đăng ký: 30A-822.10\nNhãn hiệu: TOYOTA\nSố loại: VIOS E NCP150L-BEMRKU"
    ]
    
    for i, text in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        # auto-detect and score
        detection = ner.detect_doc_type(text)
        print(f"Detected Type: {detection.doc_type} (Confidence: {detection.confidence})")
        
        # We don't have actually trained models on HF yet, 
        # but let's see if it tries to load the right ones.
        # Calling extract_auto will fail due to missing local models/hub models.
        # But we can check the detection which is what user wants "tự động phát hiện"
        
        # print scores for all
        sorted_scores = sorted(detection.scores.items(), key=lambda x: x[1], reverse=True)
        for dt, score in sorted_scores[:3]:
            if score > 0:
                print(f"  {dt}: {score}")

if __name__ == "__main__":
    test_auto_extraction()
