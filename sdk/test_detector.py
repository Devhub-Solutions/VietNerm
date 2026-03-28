"""
Test DocTypeDetector và VietNerm.extract_auto()
"""
import sys
sys.path.insert(0, "/home/ubuntu/VietNerm/sdk")

from vietnerm import DocTypeDetector, VietNerm

detector = DocTypeDetector()

SAMPLES = {
    "CCCD rõ ràng": """
        CĂN CƯỚC CÔNG DÂN
        Số định danh cá nhân: 079203030140
        Họ, chữ đệm và tên khai sinh: NGUYỄN VĂN AN
        Ngày, tháng, năm sinh: 01/01/1990
        Quê quán: Hà Nội
        Có giá trị đến: 01/01/2035
    """,
    "CCCD OCR nhiễu": """
        CAN CUOC CONG DAN
        So: 079203030140
        Ho va ten: NGUYEN VAN AN
        Ngay sinh: 01/01/1990
        Noi thuong tru: 123 Duong Lang Ha Noi
    """,
    "Giấy ra viện rõ ràng": """
        BỆNH VIỆN BẠCH MAI
        GIẤY RA VIỆN
        Họ tên người bệnh: LÊ THỊ HẰNG
        Năm sinh: 1985
        Chẩn đoán: Viêm phổi cộng đồng
        Phương pháp điều trị: Kháng sinh
        Vào viện lúc: 10/01/2024
        Ra viện lúc: 20/01/2024
        Mã số BHXH: 123456789
    """,
    "Đăng ký xe rõ ràng": """
        CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
        GIẤY ĐĂNG KÝ XE
        Chủ xe: TRẦN VĂN BÌNH
        Biển số đăng ký: 51A-123.45
        Loại xe: Xe ô tô con
        Nhãn hiệu: Toyota
        Số máy: 1NZ-FE-12345
        Số khung: JT2BF22K100123456
    """,
    "Văn bản không rõ loại": """
        Kính gửi: Ban Giám đốc
        Tôi tên là Nguyễn Văn A, xin trình bày vấn đề như sau...
    """,
    "Chỉ có tên và ngày sinh (mơ hồ)": """
        Họ và tên: Nguyễn Văn A
        Ngày sinh: 01/01/1990
        Giới tính: Nam
        Địa chỉ: Hà Nội
    """,
}

print("=" * 65)
print("TEST 1: DocTypeDetector.detect()")
print("=" * 65)
for label, text in SAMPLES.items():
    result = detector.detect(text)
    status = "✅" if result.is_confident else "❓"
    print(f"\n{status} [{label}]")
    print(f"   doc_type   : {result.doc_type}")
    print(f"   confidence : {result.confidence:.3f}")
    scores_str = "  |  ".join(f"{k}: {v:.2f}" for k, v in result.scores.items())
    print(f"   scores     : {scores_str}")

print()
print("=" * 65)
print("TEST 2: detector.detect_top_n()")
print("=" * 65)
ambiguous = SAMPLES["Chỉ có tên và ngày sinh (mơ hồ)"]
top = detector.detect_top_n(ambiguous, n=3)
print("Top 3 candidates cho văn bản mơ hồ:")
for rank, (dt, conf) in enumerate(top, 1):
    print(f"  {rank}. {dt}: {conf:.3f}")

print()
print("=" * 65)
print("TEST 3: VietNerm.extract_auto() — CCCD")
print("=" * 65)
ner = VietNerm()
result = ner.extract_auto(SAMPLES["CCCD rõ ràng"])
print(f"doc_type           : {result['doc_type']}")
print(f"detection_confidence: {result['detection_confidence']:.3f}")
print(f"detection_scores   : {result['detection_scores']}")
print(f"fields:")
for k, v in result["fields"].items():
    if v:
        print(f"   {k}: {v}")

print()
print("=" * 65)
print("TEST 4: VietNerm.extract_auto() — Giấy ra viện")
print("=" * 65)
result2 = ner.extract_auto(SAMPLES["Giấy ra viện rõ ràng"])
print(f"doc_type           : {result2['doc_type']}")
print(f"detection_confidence: {result2['detection_confidence']:.3f}")
print(f"fields:")
for k, v in result2["fields"].items():
    if v:
        print(f"   {k}: {v}")

print()
print("=" * 65)
print("TEST 5: VietNerm.extract_auto() — văn bản không rõ loại")
print("=" * 65)
result3 = ner.extract_auto(SAMPLES["Văn bản không rõ loại"])
print(f"doc_type           : {result3['doc_type']}  ← None = không detect được")
print(f"detection_confidence: {result3['detection_confidence']:.3f}")
print(f"fields             : {result3['fields']}")

print()
print("DONE")
