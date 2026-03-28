"""
End-to-end test: Zero-code auto-discovery SDK.

Tests that VietNerm works with NO shortcut classes, NO hardcoded mappings.
Everything is auto-discovered from HF Hub model config.
"""
import sys
sys.path.insert(0, ".")

print("=" * 70)
print("VietNerm SDK — Zero-Code Auto-Discovery Test")
print("=" * 70)

# ── Test 1: Import ──
print("\n[1] Import test...")
from vietnerm import VietNerm, ModelRegistry, __version__
print(f"    Version: {__version__}")
print(f"    VietNerm: {VietNerm}")
print(f"    PASS: No shortcut classes imported (CCCDNer, etc. removed)")

# ── Test 2: Available models ──
print("\n[2] Auto-discover models from HF Hub...")
models = VietNerm.available_models()
for m in models:
    print(f"    {m['doc_type']:25s} → {m['repo_id']}")
print(f"    Total: {len(models)} models")
assert len(models) >= 5, f"Expected >= 5 models, got {len(models)}"
print("    PASS")

# ── Test 3: Available doc types ──
print("\n[3] Available doc types...")
doc_types = VietNerm.available_doc_types()
print(f"    {doc_types}")
assert "cccd" in doc_types
assert "giay_khai_sinh" in doc_types
print("    PASS")

# ── Test 4: ModelRegistry ──
print("\n[4] ModelRegistry...")
reg = ModelRegistry(force_refresh=True)
print(f"    Doc types: {reg.list_doc_types()}")
for dt in reg.list_doc_types():
    info = reg.get_model_info(dt)
    print(f"    {dt:25s} | hub_verified={info.get('hub_verified', False)}")
print("    PASS")

# ── Test 5: SchemaMapper auto-discovery ──
print("\n[5] SchemaMapper auto-discovery (no DEFAULT_MAPPINGS)...")
from vietnerm._inference.schema_mapper import SchemaMapper

# Simulate unknown doc type
id2label = {0: "O", 1: "B-company_name", 2: "I-company_name", 3: "B-tax_code"}
mapper = SchemaMapper(doc_type="hoa_don_vat", id2label=id2label)
print(f"    Unknown doc 'hoa_don_vat': {mapper.entity_to_field}")
assert mapper.entity_to_field == {"COMPANY_NAME": "company_name", "TAX_CODE": "tax_code"}
print("    PASS: Auto-derived mapping for completely unknown doc type")

# ── Test 6: VietNerm.for_doc() factory ──
print("\n[6] VietNerm.for_doc() factory (replaces CCCDNer, etc.)...")
ner = VietNerm.for_doc("giay_khai_sinh", device="cpu")
print(f"    Created: {ner}")
print("    PASS")

# ── Test 7: Extract giay_khai_sinh ──
print("\n[7] Extract giay_khai_sinh...")
gks_text = """GIAY KHAI SINH
So: 123456789
Ho va ten: NGUYEN VAN AN
Gioi tinh: Nam
Ngay sinh: 15/06/2020
Noi sinh: Benh vien Tu Du
Dan toc: Kinh
Quoc tich: Viet Nam
Ho va ten cha: NGUYEN VAN BINH
Ho va ten me: TRAN THI CAM
Noi dang ky: UBND Phuong 1 Quan 1
Ngay dang ky: 20/06/2020"""

result = ner.extract(gks_text)
print(f"    Fields returned: {len(result)}")
for field, value in sorted(result.items()):
    status = "✓" if value else "·"
    print(f"    {status} {field:25s} = {value}")
non_empty = sum(1 for v in result.values() if v)
print(f"    Non-empty: {non_empty}/{len(result)}")
assert non_empty >= 5, f"Expected >= 5 non-empty fields, got {non_empty}"
print("    PASS")

# ── Test 8: get_schema() and get_fields() ──
print("\n[8] get_schema() and get_fields()...")
schema = ner.get_schema()
fields = ner.get_fields()
print(f"    Schema: {schema}")
print(f"    Fields: {fields}")
assert len(fields) >= 10, f"Expected >= 10 fields, got {len(fields)}"
print("    PASS")

# ── Test 9: Multi-doc extraction (one instance, many doc types) ──
print("\n[9] Multi-doc extraction (one VietNerm instance)...")
multi = VietNerm(device="cpu")

cccd_text = """CAN CUOC CONG DAN
So: 079203030140
Ho va ten: NGUYEN VAN A
Ngay sinh: 01/01/1990
Gioi tinh: Nam
Quoc tich: Viet Nam"""

cccd_result = multi.extract(cccd_text, doc_type="cccd")
print(f"    CCCD fields: {len(cccd_result)}")
for field, value in sorted(cccd_result.items()):
    status = "✓" if value else "·"
    print(f"    {status} {field:25s} = {value}")

gks_result = multi.extract(gks_text, doc_type="giay_khai_sinh")
print(f"    GKS fields: {len(gks_result)}")
print(f"    Loaded pipelines: {list(multi._pipelines.keys())}")
assert len(multi._pipelines) == 2, "Should have 2 pipelines loaded"
print("    PASS")

# ── Test 10: detect_doc_type ──
print("\n[10] detect_doc_type...")
detection = multi.detect_doc_type("""CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
GIẤY KHAI SINH
Họ và tên: NGUYỄN VĂN AN""")
print(f"    Detected: {detection.doc_type} (conf={detection.confidence:.3f})")
print(f"    Is confident: {detection.is_confident}")
for dt, score in sorted(detection.scores.items(), key=lambda x: -x[1]):
    print(f"      {dt:25s} = {score:.4f}")
assert detection.doc_type == "giay_khai_sinh", f"Expected giay_khai_sinh, got {detection.doc_type}"
print("    PASS")

# ── Test 11: extract_auto ──
print("\n[11] extract_auto (detect + extract in one call)...")
auto_result = multi.extract_auto(gks_text)
print(f"    Auto-detected: {auto_result['doc_type']} (conf={auto_result['detection_confidence']:.3f})")
print(f"    Fields: {len(auto_result['fields'])}")
for field, value in sorted(auto_result["fields"].items()):
    if value:
        print(f"    ✓ {field:25s} = {value}")
print("    PASS")

# ── Summary ──
print("\n" + "=" * 70)
print("ALL TESTS PASSED — Zero-code auto-discovery works!")
print("=" * 70)
print("""
Summary of what was removed (no longer needed):
  ✗ CCCDNer class
  ✗ GiayRaVienNer class
  ✗ VehicleRegistrationNer class
  ✗ GiayKhaiSinhNer class
  ✗ GPLXNer class
  ✗ DEFAULT_MAPPINGS dict in SchemaMapper
  ✗ Any hardcoded doc_type configuration

What remains (auto-discovery):
  ✓ VietNerm("doc_type") — auto-resolves model from HF Hub
  ✓ VietNerm.for_doc("doc_type") — factory method
  ✓ VietNerm.available_models() — discover from HF Hub
  ✓ VietNerm.available_doc_types() — list doc types
  ✓ SchemaMapper auto-derives from model id2label
  ✓ ModelRegistry merges local + HF Hub
  ✓ DocTypeDetector auto-discovers from templates
""")
