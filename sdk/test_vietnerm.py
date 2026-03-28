"""
Test VietNerm SDK — predict từ HuggingFace Hub models.
"""
import sys
sys.path.insert(0, "/home/ubuntu/VietNerm/sdk")

from vietnerm import VietNerm, CCCDNer, GiayRaVienNer, VehicleRegistrationNer

CCCD_TEXT = """
Căn cước công dân
Số: 079203030140
Họ và tên: NGUYỄN VĂN AN
Ngày sinh: 01/01/1990
Giới tính: Nam
Quốc tịch: Việt Nam
Quê quán: Hà Nội
Nơi thường trú: 123 Đường Láng, Đống Đa, Hà Nội
Có giá trị đến: 01/01/2035
"""

GRV_TEXT = """
GIẤY RA VIỆN
Bệnh viện: Bệnh viện Bạch Mai
Khoa: Nội tổng hợp
Mã bệnh nhân: BN-2024-001
Họ tên người bệnh: LÊ THỊ HẰNG
Ngày sinh: 15/03/1985
Giới tính: Nữ
Dân tộc: Kinh
Địa chỉ: 45 Trần Hưng Đạo, Hoàn Kiếm, Hà Nội
Ngày vào viện: 10/01/2024
Ngày ra viện: 20/01/2024
Chẩn đoán: Viêm phổi cộng đồng
Phương pháp điều trị: Kháng sinh đường tĩnh mạch
"""

VR_TEXT = """
GIẤY ĐĂNG KÝ XE
Biển số: 51A-123.45
Chủ xe: TRẦN VĂN BÌNH
Địa chỉ: 78 Nguyễn Huệ, Quận 1, TP.HCM
Nhãn hiệu: Toyota
Loại xe: Xe ô tô con
Số máy: 1NZ-FE-12345
Số khung: JT2BF22K100123456
Năm sản xuất: 2020
"""

def test_doc(label, ner, text):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"Model: {ner._resolve_model_path(ner.doc_type)}")
    print(f"{'='*60}")
    try:
        result = ner.extract(text)
        print("✅ extract() OK:")
        for k, v in result.items():
            if v:
                print(f"   {k}: {v}")

        raw = ner.extract_raw(text)
        print(f"\n✅ extract_raw() → {len(raw)} entities")
        for ent in raw:
            print(f"   [{ent['type']}] '{ent['text']}' (conf={ent['confidence']:.3f})")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()

print("VietNerm SDK — HuggingFace Hub Test")
print(f"Python: {sys.version.split()[0]}")

# Test 1: CCCD
test_doc("CCCD (Căn cước công dân)", CCCDNer(), CCCD_TEXT)

# Test 2: Giấy ra viện
test_doc("Giấy ra viện", GiayRaVienNer(), GRV_TEXT)

# Test 3: Đăng ký xe
test_doc("Đăng ký xe", VehicleRegistrationNer(), VR_TEXT)

print("\n" + "="*60)
print("DONE")
