import sys
from pathlib import Path
sys.path.insert(0, "./sdk")

from vietnerm._inference.postprocess import clean_entity_boundaries, validate_entity

# Giả lập kết quả NER từ model (đã bị lỗi như user mô tả)
raw_entities = [
    {"type": "DOB", "text": "03", "start": 0, "end": 2}, # Lỗi DOB bị cắt
    {"type": "DATE_OF_EXPIRY", "text": "30/04/2041", "start": 10, "end": 20},
    {"type": "FULL_NAME", "text": "VŨ THI BÍCH", "start": 30, "end": 41},
    {"type": "GENDER", "text": "Nữ", "start": 50, "end": 52},
    {"type": "ID_NUMBER", "text": "0320495186074", "start": 60, "end": 73}, # Giả lập có số nhưng có thể bị dính space
    {"type": "NATIONALITY", "text": "Việt Nam", "start": 80, "end": 88},
    {"type": "PLACE_OF_ORIGIN", "text": "Trần", "start": 90, "end": 94},
    {"type": "PLACE_OF_RESIDENCE", "text": "17/9/2 Trần Xã Nguyên Hãn, Vĩnh Bảo, Hải Phòng", "start": 100, "end": 146}
]

print("--- Testing Post-processing Fixes ---")
cleaned = clean_entity_boundaries(raw_entities)
for ent in cleaned:
    is_valid = validate_entity(ent["type"], ent["text"])
    print(f"{ent['type']}: '{ent['text']}' | Valid: {is_valid}")

# Kiểm tra trường hợp ID_NUMBER có dấu cách
id_with_space = "032 049 518 607"
print(f"\nTesting ID_NUMBER with spaces: '{id_with_space}'")
print(f"Valid: {validate_entity('ID_NUMBER', id_with_space)}")

# Kiểm tra trường hợp DOB chỉ có 2 số (như user bị)
dob_short = "03"
print(f"\nTesting DOB short: '{dob_short}'")
print(f"Valid: {validate_entity('DOB', dob_short)}")
