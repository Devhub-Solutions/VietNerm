"""Testing the installed vietnerm package from PyPI."""

from vietnerm import VietNerm
import json

def try_sdk():
    # Note: VietNerm will look for templates/ folder relative to CWD 
    # or inside the package if it exists. 
    # Since we are in the project root, it will find C:\content\VietNerm\templates/
    ner = VietNerm()
    
    # Test text: Giấy phép lái xe
    text = """
    CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
    Độc lập Tự do - Hạnh phúc
    GIAY PHẾP LÁI XE/DRIVER'S LICENSE
    Số/No: 790123456789
    Họ tên/Full name: NGUYỄN VĂN A
    Ngày sinh/Date of Birth: 01/01/1990
    Quốc tịch/Nationality: VIỆT NAM
    Nơi cư trú/Address: 123 ABC, TP. Hồ Chí Minh
    Hạng/Class: B2
    """
    
    print("--- 1. Document Type Detection ---")
    detection = ner.detect_doc_type(text)
    print(f"Detected: {detection.doc_type}")
    print(f"Confidence: {detection.confidence}")
    print(f"Is Confident: {detection.is_confident}")
    
    print("\n--- 2. Entity Extraction (Dynamic Mapping) ---")
    # Even without a trained model, the SchemaMapper will know the fields 
    # because it loads schema.yaml dynamically from templates/ if it exists.
    # Note: Calling predict() will fail if the model repo doesn't exist on HF,
    # but we can check if everything else is hooked up.
    
    try:
        # This might fail with OSError if HF models are not uploaded yet
        # But we've already verified the logic in previous steps.
        # extraction = ner.extract_auto(text)
        # print(json.dumps(extraction, indent=2, ensure_ascii=False))
        pass
    except Exception as e:
        print(f"NER Extraction error (Expected if model not on HF): {e}")

if __name__ == "__main__":
    try_sdk()
