"""
PhoBERT NER Pipeline cho GIẤY RA VIỆN - v2 (fixed)
═══════════════════════════════════════════════════════════════════════
Fixes so với v1:
  1. CUDA_LAUNCH_BLOCKING=1 đặt TRƯỚC mọi import torch/cuda
  2. samples_to_hf_dataset dùng explicit Features(Sequence(Value("string")))
     → tránh HF tự infer ClassLabel và encode tags thành int sai mapping
  3. tokenize_and_align_labels thêm bounds check + isinstance(tag, int) guard
  4. validate_label_ids đặt TRONG train_and_save(), trước trainer.train()
  5. warmup_steps thay warmup_ratio (deprecated)
  6. Bỏ các dòng thừa ở module scope
═══════════════════════════════════════════════════════════════════════
"""

# ── PHẢI đặt trước khi import torch / khởi tạo CUDA ──────────────────
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value
from seqeval.metrics import classification_report, f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# ═══════════════════════════════════════════
# 0. CẤU HÌNH CHUNG
# ═══════════════════════════════════════════

MODEL_NAME       = "vinai/phobert-base"
OUTPUT_DIR       = "phobert_giaravien_ner"
NUM_SAMPLES      = 2000
MAX_LENGTH       = 512
REAL_DATA_WEIGHT = 3

ENTITY_TYPES = [
    "HOSPITAL_NAME", "DEPARTMENT", "MEDICAL_CODE",
    "PATIENT_NAME", "PATIENT_DOB", "PATIENT_GENDER",
    "PATIENT_ETHNICITY", "PATIENT_OCCUPATION", "PATIENT_ADDRESS",
    "BHXH_CODE", "ADMISSION_DATE", "DISCHARGE_DATE",
    "DIAGNOSIS", "TREATMENT_METHOD", "NOTES",
]

ENTITY_TO_FIELD = {
    "HOSPITAL_NAME":      "hospital_name",
    "DEPARTMENT":         "department",
    "MEDICAL_CODE":       "medical_code",
    "PATIENT_NAME":       "patient_name",
    "PATIENT_DOB":        "patient_dob",
    "PATIENT_GENDER":     "patient_gender",
    "PATIENT_ETHNICITY":  "patient_ethnicity",
    "PATIENT_OCCUPATION": "patient_occupation",
    "PATIENT_ADDRESS":    "patient_address",
    "BHXH_CODE":          "bhxh_code",
    "ADMISSION_DATE":     "admission_date",
    "DISCHARGE_DATE":     "discharge_date",
    "DIAGNOSIS":          "diagnosis",
    "TREATMENT_METHOD":   "treatment_method",
    "NOTES":              "notes",
}


# ═══════════════════════════════════════════
# 1. DỮ LIỆU SINH SYNTHETIC
# ═══════════════════════════════════════════

HO_LIST = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý",
]
LOTEN_LIST = [
    "Văn An", "Văn Bình", "Thị Hoa", "Thị Lan", "Minh Anh", "Quốc Huy",
    "Đức Thắng", "Ngọc Hà", "Thị Mai", "Hữu Đức", "Thanh Tùng", "Thị Thu",
    "Văn Nam", "Thị Ngọc", "Bảo Châu", "Gia Huy", "Hoàng Long", "Thị Bích",
    "Văn Toàn", "Thị Hằng", "Đình Phúc", "Thị Tuyết", "Xuân Trường",
    "Diệp Anh", "Thị Hà",
]
ETHNICITY_LIST = ["Kinh", "Tày", "Thái", "Mường", "Khmer", "Hoa", "Nùng"]
OCCUPATION_LIST = [
    "Nông dân", "Công nhân", "Giáo viên", "Bác sĩ", "Kỹ sư", "Buôn bán",
    "Học sinh", "Sinh viên", "Hưu trí", "Nội trợ", "Người già",
    "Lao động tự do",
]
PROVINCE_LIST = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "An Giang", "Khánh Hòa", "Nghệ An",
    "Thanh Hóa", "Lâm Đồng", "Quảng Nam", "Quảng Ninh", "Huế",
    "Bắc Giang", "Thái Nguyên", "Gia Lai", "Đắk Lắk", "Kiên Giang",
]
DISTRICT_LIST = [
    "Quận 1", "Quận 3", "Quận 7", "Quận Bình Thạnh", "Huyện Củ Chi",
    "Huyện Lâm Hà", "Huyện Quế Sơn", "Thành phố Hạ Long",
    "Quận Hoàn Kiếm", "Huyện Gia Lâm", "Huyện Đức Trọng",
]
WARD_LIST = [
    "Phường Tân Phú", "Xã Tân Hà", "Xã Quế Xuân 2", "Phường Cao Thắng",
    "Thôn Phủ Mỹ", "Phường Lộc Sơn", "Phường 3", "Xã Lộc Tân",
]
STREET_LIST = [
    "Trần Hưng Đạo", "Nguyễn Huệ", "Lê Lợi", "Phan Chu Trinh",
    "Hai Bà Trưng", "Đinh Tiên Hoàng", "Cách Mạng Tháng 8",
    "Nam Kỳ Khởi Nghĩa", "H. Tấn Phát",
]

HOSPITAL_LIST = [
    ("Bệnh viện Đa khoa tỉnh Lâm Đồng",  "Sở Y tế Lâm Đồng"),
    ("Bệnh viện Đa khoa Vĩnh Đức",        "Sở Y tế Quảng Nam"),
    ("Bệnh viện Đa khoa tỉnh Quảng Ninh", "Sở Y tế Quảng Ninh"),
    ("Bệnh viện Đa khoa tỉnh Nghệ An",    "Sở Y tế Nghệ An"),
    ("Bệnh viện Đa khoa Trung ương Huế",  "Bộ Y tế"),
    ("Bệnh viện Nhân dân 115",            "Sở Y tế TP. Hồ Chí Minh"),
    ("Bệnh viện Bạch Mai",                "Bộ Y tế"),
    ("Bệnh viện Chợ Rẫy",                 "Bộ Y tế"),
    ("Bệnh viện Đa khoa Gia Lai",         "Sở Y tế Gia Lai"),
    ("Bệnh viện Đa khoa tỉnh Bình Dương", "Sở Y tế Bình Dương"),
]

DEPARTMENT_LIST = [
    "Khoa Nội A", "Khoa Nội Tổng hợp",
    "Khoa Ngoại Chấn thương Chỉnh hình", "Khoa Nhi",
    "Khoa Hồi sức tích cực", "Khoa Tim mạch", "Khoa Thần kinh",
    "Khoa Sản", "Khoa Tai Mũi Họng", "Khoa Mắt", "Khoa Da liễu",
    "Khoa Tiêu hóa", "Khoa Ung bướu", "Khoa Cấp cứu", "Khoa Nội thận",
]

DIAGNOSIS_LIST = [
    ("I63",   "Nhồi máu não"),
    ("J00",   "Viêm họng cấp"),
    ("M51.1", "Bệnh đĩa đệm cột sống thắt lưng có tổn thương rễ tủy"),
    ("I10",   "Tăng huyết áp nguyên phát"),
    ("E11",   "Đái tháo đường type 2"),
    ("J18.9", "Viêm phổi không xác định"),
    ("K29.5", "Viêm dạ dày mạn tính"),
    ("N18",   "Bệnh thận mạn"),
    ("S82.0", "Gãy xương bánh chè"),
    ("C34",   "Ung thư phế quản và phổi"),
    ("G40",   "Động kinh"),
    ("A91",   "Sốt xuất huyết Dengue"),
    ("I50",   "Suy tim"),
    ("B34.9", "Nhiễm virus không xác định"),
]

TREATMENT_LIST = [
    "Nội khoa: kháng sinh, bù nước điện giải",
    "Phẫu thuật nội soi lấy nhân đệm qua lỗ liên hợp dưới màn hình tăng sáng",
    "Nội khoa: hạ áp, lợi tiểu, chống đông",
    "Truyền dịch, kháng sinh tiêm tĩnh mạch, hạ sốt",
    "Phẫu thuật kết xương nẹp vít",
    "Hóa trị liệu phác đồ FOLFOX",
    "Vật lý trị liệu, thuốc chống viêm không steroid",
    "Insulin, metformin, kiểm soát đường huyết",
    "Thuốc chống động kinh, valproate 500mg",
    "Cerebrolysin 10ml, Tanagnil 1g, Aspirin 80mg, Atorvastatin 20mg",
]

NOTE_LIST = [
    "Tái khám sau 7 ngày",
    "Khám lại khi có bất thường",
    "Uống thuốc theo đơn, thay băng hằng ngày tại trạm y tế cơ sở",
    "Cắt chỉ sau 10 ngày, tái khám định kỳ",
    "Tiếp tục điều trị ngoại trú theo đơn",
    "Nghỉ ngơi tại nhà, hạn chế vận động nặng",
    "Khám lại tại phòng khám ngoại trú, mang theo kết quả xét nghiệm",
]

OCR_NOISE_TOKENS = [
    "Star", "It", "AND", "OF", "THE", "MS:",
    "REFITTED", "ZORX", "riston", "Therming",
    "windswy", "Sentifier", "Cesm", "Emok",
    "Contrước", "Trong", "Thuật", "CỔ PHẦN",
    "j5072810750", "Swoodbản",
]

LABEL_TEXTS: Dict[str, List[str]] = {
    "HOSPITAL_NAME": ["BỆNH VIỆN ĐA KHOA", "Bệnh viện", "BỆNH VIỆN"],
    "DEPARTMENT": ["Khoa:", "Khoa", "KHOA:"],
    "MEDICAL_CODE": [
        "Số lưu trữ:", "Mã y tế:", "Số lưu:", "MS:", "Mã Y tế:", "Số lưu trữ",
    ],
    "PATIENT_NAME": [
        "Họ tên người bệnh:", "- Họ tên người bệnh:",
        "Họ và tên người bệnh:", "Họ tên bệnh nhân:", "HỌ TÊN NGƯỜI BỆNH:",
    ],
    "PATIENT_DOB": [
        "Năm sinh:", "Ngày sinh:", "Ngày/tháng/năm sinh:",
        "Sinh ngày:", "- Năm sinh:",
    ],
    "PATIENT_GENDER": ["Giới tính:", "Nam/Nữ:", "Giới tính", "Sex:"],
    "PATIENT_ETHNICITY": ["Dân tộc:", "- Dân tộc:", "Dân tộc"],
    "PATIENT_OCCUPATION": ["Nghề nghiệp:", "Nghề nghiệp", "- Nghề nghiệp:"],
    "PATIENT_ADDRESS": ["Địa chỉ:", "- Địa chỉ:", "Địa chỉ", "Nơi ở:"],
    "BHXH_CODE": [
        "Mã Số BHXH/Thẻ BHYT Số:", "- Mã Số BHXH/Thẻ BHYT Số:",
        "MÃ SỔ BHXH/THẺ BHYT SỐ:", "Mã số BHXH:", "Thẻ BHYT:", "Mã BHYT:",
    ],
    "ADMISSION_DATE": [
        "Vào viện lúc:", "- Vào viện lúc", "Ngày vào viện:", "Vào viện:", "Nhập viện:",
    ],
    "DISCHARGE_DATE": [
        "Ra viện lúc:", "- Ra viện lúc", "Ngày ra viện:", "Ra viện:", "Xuất viện:",
    ],
    "DIAGNOSIS": [
        "Chẩn đoán:", "- Chẩn đoán:", "CHẨN ĐOÁN:",
        "Chẩn đoán ra viện:", "Chản đoán:", "Chần đoán:",
    ],
    "TREATMENT_METHOD": [
        "Phương pháp điều trị:", "- Phương pháp điều trị:",
        "Điều trị:", "Thuật Phương pháp điều trị:",
    ],
    "NOTES": ["Ghi chú:", "- Ghi chú:", "GHI CHÚ:", "Tái khám:", "Tái khám sau:"],
}


# ═══════════════════════════════════════════
# 1b. HELPER FUNCTIONS
# ═══════════════════════════════════════════

def random_date_grv(start_year: int = 1940, end_year: int = 2020) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    d     = start + timedelta(days=random.randint(0, (end - start).days))
    if random.random() < 0.3:
        return d.strftime("%Y")
    return d.strftime("%d/%m/%Y")


def random_admission_discharge() -> Tuple[str, str]:
    admit = datetime(2020, 1, 1) + timedelta(days=random.randint(0, 1500))
    disch = admit + timedelta(days=random.randint(1, 20))
    formats = [
        ("{h} giờ {m} phút, ngày {d} tháng {mo} năm {y}",
         "{h} giờ {m} phút, ngày {d} tháng {mo} năm {y}"),
        ("{h} giờ {m} ngày {d} tháng {mo} năm {y}",
         "{h} giờ {m} ngày {d} tháng {mo} năm {y}"),
        ("{d}/{mo}/{y}", "{d}/{mo}/{y}"),
        ("ngày {d} tháng {mo} năm {y}", "ngày {d} tháng {mo} năm {y}"),
    ]
    fmt_admit, fmt_disch = random.choice(formats)

    def _fmt(dt: datetime, fmt: str) -> str:
        return fmt.format(
            h=str(random.randint(6, 22)),
            m=str(random.randint(0, 59)).zfill(2),
            d=str(dt.day), mo=str(dt.month), y=str(dt.year),
        )
    return _fmt(admit, fmt_admit), _fmt(disch, fmt_disch)


def random_bhxh_code() -> str:
    styles = [
        lambda: "GD" + "".join([str(random.randint(0, 9)) for _ in range(13)]),
        lambda: ("HS " + " ".join([
            "".join([str(random.randint(0, 9)) for _ in range(random.randint(3, 5))])
            for _ in range(3)
        ])),
        lambda: ("GD " + str(random.randint(1, 9)) + " "
                 + str(random.randint(10, 99)) + " "
                 + "".join([str(random.randint(0, 9)) for _ in range(11)])),
    ]
    return random.choice(styles)()


def random_medical_code() -> str:
    year = random.randint(20, 24)
    seq  = random.randint(1000, 99999)
    return f"{year}.{seq:06d}"


def random_name() -> str:
    ho     = random.choice(HO_LIST)
    lo_ten = random.choice(LOTEN_LIST)
    full   = f"{ho} {lo_ten}"
    style  = random.random()
    if style < 0.4:
        return full.upper()
    elif style < 0.7:
        return full
    return full.lower()


def random_address() -> str:
    styles = [
        lambda: (f"Thôn {random.choice(['Phủ Mỹ','Trung','An','Tân'])},"
                 f" {random.choice(WARD_LIST)}, {random.choice(DISTRICT_LIST)},"
                 f" {random.choice(PROVINCE_LIST)}"),
        lambda: (f"Số {random.randint(1, 300)} {random.choice(STREET_LIST)},"
                 f" {random.choice(WARD_LIST)}, {random.choice(DISTRICT_LIST)},"
                 f" {random.choice(PROVINCE_LIST)}"),
        lambda: (f"{random.choice(WARD_LIST)}, {random.choice(DISTRICT_LIST)},"
                 f" {random.choice(PROVINCE_LIST)}"),
        lambda: (f"{random.randint(1,999)}/{random.randint(1,99)}/{random.randint(1,30)}"
                 f" {random.choice(STREET_LIST)}, KP{random.randint(1,5)},"
                 f" {random.choice(WARD_LIST)}, {random.choice(PROVINCE_LIST)}"),
    ]
    return random.choice(styles)()


def add_ocr_noise(text: str, prob: float = 0.25) -> str:
    if random.random() < prob:
        noise = random.choice(OCR_NOISE_TOKENS)
        if random.random() < 0.5:
            return noise + " " + text
        return text + " " + noise
    return text


def generate_records(n: int = NUM_SAMPLES) -> List[Dict]:
    records = []
    for _ in range(n):
        hosp_name, dept_mgmt = random.choice(HOSPITAL_LIST)
        diag_code, diag_name = random.choice(DIAGNOSIS_LIST)
        if random.random() < 0.4:
            ec, en    = random.choice(DIAGNOSIS_LIST)
            diagnosis = f"{diag_name} ({diag_code}); {en} ({ec})"
        else:
            diagnosis = f"{diag_name} ({diag_code})"
        admission, discharge = random_admission_discharge()
        rec = {
            "hospital_name":      hosp_name,
            "dept_mgmt":          dept_mgmt,
            "department":         random.choice(DEPARTMENT_LIST),
            "medical_code":       random_medical_code(),
            "patient_name":       random_name(),
            "patient_dob":        random_date_grv(),
            "patient_gender":     random.choice(["Nam", "Nữ"]),
            "patient_ethnicity":  random.choice(ETHNICITY_LIST),
            "patient_occupation": random.choice(OCCUPATION_LIST),
            "patient_address":    random_address(),
            "bhxh_code":          random_bhxh_code(),
            "admission_date":     admission,
            "discharge_date":     discharge,
            "diagnosis":          diagnosis,
            "treatment_method":   random.choice(TREATMENT_LIST),
            "notes":              random.choice(NOTE_LIST),
        }
        records.append(rec)
    return records


# ═══════════════════════════════════════════
# 2. BUILD TEXT + SPAN MAP
# ═══════════════════════════════════════════

def record_to_text_and_spans(
    rec: Dict,
) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    spans: List[Tuple[int, int, str, str]] = []

    header_variants = [
        [
            rec["dept_mgmt"].upper(),
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            rec["hospital_name"].upper(),
            "Độc lập - Tự do - Hạnh phúc",
        ],
        [
            rec["dept_mgmt"],
            rec["hospital_name"],
            "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
            "Độc lập - Tự do - Hạnh phúc",
        ],
    ]
    header_lines = [add_ocr_noise(h, 0.35) for h in random.choice(header_variants)]

    mc_label   = random.choice(LABEL_TEXTS["MEDICAL_CODE"])
    dept_label = random.choice(LABEL_TEXTS["DEPARTMENT"])
    meta_lines = [
        f"{mc_label} {rec['medical_code']}",
        f"{dept_label} {rec['department']}",
        "GIẤY RA VIỆN",
    ]
    if random.random() < 0.3:
        random.shuffle(meta_lines[:2])

    SEPS = [": ", " ", "\n", "/ ", ":\n"]
    patient_fields = [
        ("PATIENT_NAME",       rec["patient_name"]),
        ("PATIENT_DOB",        rec["patient_dob"]),
        ("PATIENT_GENDER",     rec["patient_gender"]),
        ("PATIENT_ETHNICITY",  rec["patient_ethnicity"]),
        ("PATIENT_OCCUPATION", rec["patient_occupation"]),
        ("BHXH_CODE",          rec["bhxh_code"]),
        ("PATIENT_ADDRESS",    rec["patient_address"]),
    ]
    merged_gender_ethn = random.random() < 0.45

    clinical_fields = [
        ("ADMISSION_DATE",   rec["admission_date"]),
        ("DISCHARGE_DATE",   rec["discharge_date"]),
        ("DIAGNOSIS",        rec["diagnosis"]),
        ("TREATMENT_METHOD", rec["treatment_method"]),
        ("NOTES",            rec["notes"]),
    ]

    patient_lines = []
    skip_next     = False
    for etype, value in patient_fields:
        if skip_next:
            skip_next = False
            continue
        label_text = random.choice(LABEL_TEXTS[etype])
        sep        = random.choice(SEPS)
        if etype == "PATIENT_GENDER" and merged_gender_ethn:
            ethn_label = random.choice(LABEL_TEXTS["PATIENT_ETHNICITY"])
            line = (f"{label_text}{sep}{rec['patient_gender']}"
                    f"  {ethn_label}{sep}{rec['patient_ethnicity']}")
            patient_lines.append((line, [
                (etype,               label_text, rec["patient_gender"]),
                ("PATIENT_ETHNICITY", ethn_label, rec["patient_ethnicity"]),
            ]))
            skip_next = True
            continue
        patient_lines.append((f"{label_text}{sep}{value}", [(etype, label_text, value)]))

    clinical_lines = []
    for etype, value in clinical_fields:
        label_text = random.choice(LABEL_TEXTS[etype])
        sep        = random.choice(SEPS)
        clinical_lines.append((f"{label_text}{sep}{value}", [(etype, label_text, value)]))

    noisy_patient  = [(add_ocr_noise(l, 0.25), c) for l, c in patient_lines]
    noisy_clinical = [(add_ocr_noise(l, 0.20), c) for l, c in clinical_lines]

    all_text_lines = (
        header_lines
        + meta_lines
        + [l for l, _ in noisy_patient]
        + [l for l, _ in noisy_clinical]
    )
    full_text = "\n".join(all_text_lines)

    all_configs = [(l, c) for l, c in noisy_patient + noisy_clinical]
    for _, configs in all_configs:
        for etype, label_text, value in configs:
            label_start = full_text.find(label_text)
            if label_start != -1:
                spans.append((label_start, label_start + len(label_text), etype, "LABEL"))
            search_from = label_start + len(label_text) if label_start != -1 else 0
            value_start = full_text.find(value, search_from)
            if value_start != -1:
                spans.append((value_start, value_start + len(value), etype, "VALUE"))

    for variant in [rec["hospital_name"], rec["hospital_name"].upper()]:
        v_start = full_text.find(variant)
        if v_start != -1:
            spans.append((v_start, v_start + len(variant), "HOSPITAL_NAME", "VALUE"))
            break

    for d_label in LABEL_TEXTS["DEPARTMENT"]:
        d_start = full_text.find(d_label)
        if d_start != -1:
            spans.append((d_start, d_start + len(d_label), "DEPARTMENT", "LABEL"))
    dept_val_start = full_text.find(rec["department"])
    if dept_val_start != -1:
        spans.append((dept_val_start, dept_val_start + len(rec["department"]),
                      "DEPARTMENT", "VALUE"))

    return full_text, spans


# ═══════════════════════════════════════════
# 3. TOKENIZE + GÁN NHÃN BIO
# ═══════════════════════════════════════════

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    flat   = text.replace("\n", " ")
    result = []
    pos    = 0
    for tok in flat.split(" "):
        if not tok:
            pos += 1
            continue
        start = flat.find(tok, pos)
        if start == -1:
            pos += len(tok) + 1
            continue
        end = start + len(tok)
        result.append((tok, start, end))
        pos = end + 1
    return result


def assign_bio_labels(
    tokens_with_pos: List[Tuple[str, int, int]],
    spans: List[Tuple[int, int, str, str]],
) -> List[str]:
    labels       = ["O"] * len(tokens_with_pos)
    sorted_spans = sorted(spans, key=lambda x: (0 if x[3] == "VALUE" else 1, x[0]))
    for span_start, span_end, etype, role in sorted_spans:
        tag_name      = f"{etype}_{role}"
        first_in_span = True
        for i, (_, tok_start, tok_end) in enumerate(tokens_with_pos):
            if tok_end <= span_start or tok_start >= span_end:
                continue
            if labels[i] == "O" or (role == "VALUE" and labels[i].endswith("_LABEL")):
                labels[i]     = f"B-{tag_name}" if first_in_span else f"I-{tag_name}"
                first_in_span = False
    return labels


def build_labeled_samples(records: List[Dict]) -> List[Dict]:
    samples = []
    for rec in records:
        text, spans     = record_to_text_and_spans(rec)
        tokens_with_pos = tokenize_with_offsets(text)
        if not tokens_with_pos:
            continue
        tokens = [t for t, _, _ in tokens_with_pos]
        labels = assign_bio_labels(tokens_with_pos, spans)
        samples.append({"tokens": tokens, "ner_tags": labels})
    return samples


# ═══════════════════════════════════════════
# 4. REAL DATA SUPPORT
# ═══════════════════════════════════════════

def convert_real_annotation(annotation: Dict) -> Optional[Dict]:
    text     = annotation.get("text", "")
    entities = annotation.get("entities", [])
    if not text:
        return None
    tokens_with_pos = tokenize_with_offsets(text)
    if not tokens_with_pos:
        return None
    spans = []
    for ent in entities:
        start = ent.get("start", -1)
        end   = ent.get("end", -1)
        label = ent.get("label", "")
        if start < 0 or end < 0 or not label:
            continue
        if label not in ENTITY_TYPES:
            print(f"[WARN] Unknown entity type: '{label}' - bỏ qua")
            continue
        spans.append((start, end, label, "VALUE"))
    tokens = [t for t, _, _ in tokens_with_pos]
    labels = assign_bio_labels(tokens_with_pos, spans)
    return {"tokens": tokens, "ner_tags": labels}


def load_real_data(json_path: str) -> List[Dict]:
    if not os.path.exists(json_path):
        print(f"[INFO] File real data không tìm thấy: {json_path}")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    samples = []
    for ann in annotations:
        sample = convert_real_annotation(ann)
        if sample:
            samples.append(sample)
    print(f"[INFO] Loaded {len(samples)} real samples từ {json_path}")
    return samples


# ═══════════════════════════════════════════
# 5. HuggingFace DATASET
# ═══════════════════════════════════════════

def samples_to_hf_dataset(samples: List[Dict]) -> Dataset:
    """
    [FIX] Dùng explicit Features với Sequence(Value("string")) để HF
    KHÔNG tự infer ClassLabel và encode ner_tags thành int sai mapping.
    Nếu HF tự encode → label_id nhận integer với mapping khác label2id
    → CUDA index-out-of-bounds assert trong loss computation.
    """
    features = Features({
        "tokens":   Sequence(Value("string")),
        "ner_tags": Sequence(Value("string")),
    })
    return Dataset.from_list(
        [{"tokens": s["tokens"], "ner_tags": s["ner_tags"]} for s in samples],
        features=features,
    )


def get_label_list(samples: List[Dict]) -> List[str]:
    tag_set = {"O"}
    for s in samples:
        for t in s["ner_tags"]:
            tag_set.add(t)
    return ["O"] + sorted(tag_set - {"O"})


# ═══════════════════════════════════════════
# 6. TOKENIZE & ALIGN LABELS
# ═══════════════════════════════════════════

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """
    [FIX] Thêm:
      - isinstance(tag, int) guard: nếu HF vẫn encode tags → convert về string
      - bounds check: label_id ngoài [0, num_labels) → fallback về O
      - assert cuối batch để bắt lỗi sớm trước khi vào GPU
    """
    num_labels = len(label2id)
    id_to_label_keys = list(label2id.keys())   # dùng cho int→str fallback

    all_input_ids, all_attention_mask, all_labels = [], [], []

    for i, tokens in enumerate(examples["tokens"]):
        ner_tags  = examples["ner_tags"][i]
        input_ids = [tokenizer.cls_token_id]
        labels    = [-100]   # CLS

        for text, tag in zip(tokens, ner_tags):
            # Guard: nếu tag là int (HF tự encode), chuyển lại về string
            if isinstance(tag, int):
                tag = id_to_label_keys[tag] if tag < len(id_to_label_keys) else "O"

            word_tokens = tokenizer.encode(text, add_special_tokens=False)
            if not word_tokens:
                continue

            label_id = label2id.get(tag, label2id["O"])

            # Bounds check
            if not (0 <= label_id < num_labels):
                print(f"[WARN] label_id={label_id} ngoài biên [{num_labels}], "
                      f"tag='{tag}' → fallback O")
                label_id = label2id["O"]

            input_ids.extend(word_tokens)
            labels.append(label_id)
            labels.extend([-100] * (len(word_tokens) - 1))

        input_ids.append(tokenizer.sep_token_id)
        labels.append(-100)   # SEP

        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels    = labels[:MAX_LENGTH]

        padding_len = MAX_LENGTH - len(input_ids)
        mask        = [1] * len(input_ids) + [0] * padding_len
        input_ids  += [tokenizer.pad_token_id] * padding_len
        labels     += [-100] * padding_len

        # Sanity assert — bắt lỗi trước khi vào GPU
        bad = [l for l in labels if l != -100 and not (0 <= l < num_labels)]
        assert not bad, (
            f"Sample {i}: label out of range! bad values={bad}, num_labels={num_labels}"
        )

        all_input_ids.append(input_ids)
        all_attention_mask.append(mask)
        all_labels.append(labels)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "labels":         all_labels,
    }


# ═══════════════════════════════════════════
# 7. METRICS
# ═══════════════════════════════════════════

def compute_metrics(p, id2label):
    preds, labels = p
    preds         = np.argmax(preds, axis=-1)
    true_labels, true_preds = [], []

    for pred_seq, lab_seq in zip(preds, labels):
        cur_l, cur_p = [], []
        for p_id, l_id in zip(pred_seq, lab_seq):
            if l_id == -100:
                continue
            cur_l.append(id2label[l_id])
            cur_p.append(id2label[p_id])
        true_labels.append(cur_l)
        true_preds.append(cur_p)

    overall_f1 = f1_score(true_labels, true_preds)

    try:
        report     = classification_report(true_labels, true_preds, output_dict=True)
        per_entity = {}
        for k, v in report.items():
            if isinstance(v, dict) and "_VALUE" in k and k.startswith("B-"):
                etype              = k[2:].replace("_VALUE", "")
                per_entity[etype] = v.get("f1-score", 0.0)
        if per_entity:
            print("\n[Per-entity F1 — VALUE only]")
            for etype, score in sorted(per_entity.items()):
                print(f"  {etype:25s}: {score:.4f}")
    except Exception:
        pass

    return {"f1": overall_f1}


# ═══════════════════════════════════════════
# 8. VALIDATORS & POST-PROCESSING
# ═══════════════════════════════════════════

GENDER_NORMALIZE = {
    "NỮ": "Nữ", "NU": "Nữ", "NO": "Nữ",
    "N0": "Nữ", "NƯ": "Nữ", "NỪ": "Nữ",
    "NAM": "Nam",
}

VALIDATORS: Dict[str, callable] = {
    "PATIENT_DOB":      lambda x: bool(re.match(r'^\d{2}/\d{2}/\d{4}$|^\d{4}$', x.strip())),
    "ADMISSION_DATE":   lambda x: bool(re.search(r'\d{1,2}.*\d{4}', x.strip())),
    "DISCHARGE_DATE":   lambda x: bool(re.search(r'\d{1,2}.*\d{4}', x.strip())),
    "PATIENT_GENDER":   lambda x: (
        GENDER_NORMALIZE.get(x.strip().upper(), x.strip()) in ("Nam", "Nữ")
    ),
    "PATIENT_ETHNICITY": lambda x: 1 <= len(x.split()) <= 3,
    "PATIENT_NAME":     lambda x: (
        2 <= len(x.split()) <= 7 and not re.search(r'\d', x)
    ),
    "BHXH_CODE":        lambda x: len(re.sub(r'\s', '', x)) >= 8,
    "MEDICAL_CODE":     lambda x: bool(re.search(r'\d', x)),
    "DIAGNOSIS":        lambda x: len(x.strip()) >= 5,
    "TREATMENT_METHOD": lambda x: len(x.strip()) >= 5,
}


def ner_predict(raw_text: str, tokenizer, model, id2label: Dict) -> List[Dict]:
    tokens_with_pos = tokenize_with_offsets(raw_text)
    if not tokens_with_pos:
        return []

    tokens                 = [t for t, _, _ in tokens_with_pos]
    input_ids              = [tokenizer.cls_token_id]
    token_to_subword_start = []

    for tok in tokens:
        word_tokens = tokenizer.encode(tok, add_special_tokens=False)
        if not word_tokens:
            token_to_subword_start.append(None)
            continue
        token_to_subword_start.append(len(input_ids))
        input_ids.extend(word_tokens)

    input_ids.append(tokenizer.sep_token_id)
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]

    device = next(model.parameters()).device
    inputs = torch.tensor([input_ids]).to(device)

    model.eval()
    with torch.no_grad():
        outputs  = model(inputs)

    pred_ids     = outputs.logits.argmax(-1)[0].tolist()
    token_labels = []
    for sw_start in token_to_subword_start:
        if sw_start is None or sw_start >= len(pred_ids):
            token_labels.append("O")
        else:
            token_labels.append(id2label.get(pred_ids[sw_start], "O"))

    results     = []
    current_ent: Optional[Dict] = None

    for tok, label in zip(tokens, token_labels):
        if "_VALUE" not in label:
            if current_ent:
                results.append(current_ent)
                current_ent = None
            continue

        parts = label.split("-", 1)
        if len(parts) != 2:
            if current_ent:
                results.append(current_ent)
                current_ent = None
            continue

        prefix, tag_name = parts
        etype            = tag_name.replace("_VALUE", "")

        if prefix == "B":
            if current_ent:
                results.append(current_ent)
            current_ent = {"type": etype, "text": tok}
        elif prefix == "I":
            if current_ent and current_ent["type"] == etype:
                current_ent["text"] += " " + tok
            else:
                if current_ent:
                    results.append(current_ent)
                current_ent = {"type": etype, "text": tok}

    if current_ent:
        results.append(current_ent)

    cleaned    = []
    seen_types = set()
    for ent in results:
        t    = ent["type"]
        text = ent["text"].strip()
        if t in seen_types:
            continue
        if t == "PATIENT_GENDER":
            text = GENDER_NORMALIZE.get(text.upper(), text)
        if t == "PATIENT_NAME":
            text = text.title()
        if t in VALIDATORS and not VALIDATORS[t](text):
            print(f"[SKIP] {t}: '{text}' không qua validator")
            continue
        cleaned.append({"type": t, "text": text})
        seen_types.add(t)

    return cleaned


def ents_to_grv_json(entities: List[Dict]) -> Dict:
    out = {field: "" for field in ENTITY_TO_FIELD.values()}
    for ent in entities:
        field = ENTITY_TO_FIELD.get(ent["type"])
        if field:
            out[field] = ent["text"]
    return out


# ═══════════════════════════════════════════
# 9. VALIDATE HELPER
# ═══════════════════════════════════════════

def validate_label_ids(dataset, num_labels: int, name: str = "dataset"):
    """Kiểm tra toàn bộ label IDs trước khi đưa vào GPU."""
    bad_count = 0
    for row in dataset:
        for l in row["labels"]:
            if l != -100 and not (0 <= l < num_labels):
                bad_count += 1
    if bad_count:
        raise ValueError(
            f"[{name}] Có {bad_count} label IDs ngoài [0, {num_labels - 1}]!"
        )
    print(f"[{name}] OK — tất cả label IDs hợp lệ (num_labels={num_labels}).")


# ═══════════════════════════════════════════
# 10. TRAINING PIPELINE
# ═══════════════════════════════════════════

def train_and_save(
    records: List[Dict],
    real_data_path: Optional[str] = None,
):
    print("==> Building labeled samples (synthetic)...")
    samples = build_labeled_samples(records)
    print(f"    Synthetic: {len(samples)} samples")

    if real_data_path:
        real_samples = load_real_data(real_data_path)
        if real_samples:
            samples = samples + real_samples * REAL_DATA_WEIGHT
            print(f"    Real (x{REAL_DATA_WEIGHT}): {len(real_samples) * REAL_DATA_WEIGHT} samples")

    print(f"    Total: {len(samples)} samples")
    random.shuffle(samples)

    n       = len(samples)
    train_s = samples[:int(0.8 * n)]
    valid_s = samples[int(0.8 * n):int(0.9 * n)]

    label_list = get_label_list(samples)
    label2id   = {l: i for i, l in enumerate(label_list)}
    id2label   = {i: l for i, l in enumerate(label_list)}
    print(f"==> Labels ({len(label_list)}): {label_list}")

    print("==> Loading tokenizer & PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print("==> Tokenizing datasets...")
    fn_kwargs = {"tokenizer": tokenizer, "label2id": label2id}
    train_tok = samples_to_hf_dataset(train_s).map(
        tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True
    )
    valid_tok = samples_to_hf_dataset(valid_s).map(
        tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True
    )

    # [FIX] Validate label IDs TRƯỚC khi đưa vào GPU
    print("==> Validating label IDs...")
    validate_label_ids(train_tok, len(label_list), "train")
    validate_label_ids(valid_tok, len(label_list), "valid")

    print("==> Training...")
    args = TrainingArguments(
        OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=7,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        warmup_steps=100,   # [FIX] warmup_ratio deprecated → warmup_steps
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
        json.dump({"id2label": id2label, "label2id": label2id},
                  f, ensure_ascii=False, indent=2)

    print(f"==> Saving model → {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("==> Done!")
    return tokenizer, model, id2label


def load_model(output_dir: str):
    print(f"==> Loading model from {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(output_dir)
    id2label  = {int(k): v for k, v in model.config.id2label.items()}
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)
    model.eval()
    print(f"==> Loaded! Device: {device}")
    return tokenizer, model, id2label


# ═══════════════════════════════════════════
# 11. DEBUG & EVALUATION HELPERS
# ═══════════════════════════════════════════

def debug_sample(rec: Dict):
    text, spans     = record_to_text_and_spans(rec)
    tokens_with_pos = tokenize_with_offsets(text)
    labels          = assign_bio_labels(tokens_with_pos, spans)

    print("=" * 70)
    print("TEXT:\n" + text)
    print("\nTOKEN ALIGNMENT (non-O only):")
    for (tok, _, _), lab in zip(tokens_with_pos, labels):
        if lab != "O":
            print(f"  [{lab:48s}] '{tok}'")

    value_tags = {
        lab.replace("B-", "").replace("_VALUE", "")
        for lab in labels if "_VALUE" in lab and lab.startswith("B-")
    }
    missing = [e for e in ENTITY_TYPES if e not in value_tags
               and e not in ("HOSPITAL_NAME", "DEPARTMENT", "MEDICAL_CODE")]
    if missing:
        print(f"  [WARN] Missing VALUE tags: {missing}")
    print("=" * 70)
    return text, spans, labels


def evaluate_sample(raw_text: str, ground_truth: Dict, tokenizer, model, id2label: Dict):
    entities = ner_predict(raw_text, tokenizer, model, id2label)
    result   = ents_to_grv_json(entities)

    print("\n" + "=" * 78)
    print(f"{'FIELD':<25} {'PREDICTED':<25} {'EXPECTED':<25} MATCH")
    print("-" * 78)
    correct = 0
    total   = len([v for v in ground_truth.values() if v])
    for field, expected in ground_truth.items():
        if not expected:
            continue
        predicted = result.get(field, "")
        match     = "✓" if predicted.lower() == expected.lower() else "✗"
        if match == "✓":
            correct += 1
        print(f"{field:<25} {predicted:<25} {expected:<25} {match}")
    print("-" * 78)
    print(f"Accuracy: {correct}/{total} = {correct / max(total, 1) * 100:.1f}%")
    return result


# ═══════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════

def main():
    real_data_path = "real_grv_annotations.json"

    if (os.path.exists(OUTPUT_DIR)
            and os.path.isfile(os.path.join(OUTPUT_DIR, "config.json"))):
        tokenizer, model, id2label = load_model(OUTPUT_DIR)
    else:
        print("==> Không tìm thấy model đã lưu. Bắt đầu training...")
        records = generate_records(NUM_SAMPLES)

        print("\n[DEBUG] Kiểm tra data generation (2 samples):")
        debug_sample(records[0])
        debug_sample(records[1])

        tokenizer, model, id2label = train_and_save(
            records,
            real_data_path=real_data_path if os.path.exists(real_data_path) else None,
        )

    print("\n" + "=" * 70)
    print("==> Demo predict với OCR text thực tế")

    sample1 = """SỞ Y TẾ LÂM ĐỒNG
BỆNH VIỆN ĐA KHOA TỈNH LÂM ĐỒNG
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc
Số lưu: 2023.031456
Khoa: Khoa Nội A
GIẤY RA VIỆN
- Họ tên người bệnh: lê thị hằng trị
Năm sinh: 23/10/1952
- Dân tộc: Kinh
Nghề nghiệp: người già
- Mã Số BHXH/Thẻ BHYT Số: GD4686822576364-68510
- Địa chỉ: Liên Trung, Xã Tân Hà, Huyện Lâm Hà, Lâm Đồng
- Vào viện lúc 19 giờ 20 phút, ngày 2 tháng 4 năm 2023
- Ra viện lúc 10 giờ 0 phút, ngày 7 tháng 4 năm 2023
- Chẩn đoán: I63-Nhồi máu não; Chóng mặt; Rối loạn Lipid máu
- Phương pháp điều trị: Nội KHOA (Cerebrolysin 10ml, Tanagnil 1g, Aspirin 80mg)
- Ghi chú: KHÁM VÀ ĐIỀU TRỊ TIẾP TẠI NƠI ĐĂNG KÝ KHÁM CHỮA BỆNH BAN ĐẦU"""

    sample2 = """SỞ Y TẾ QUẢNG NINH
BỆNH VIỆN ĐA KHOA TỈNH
Độc lập - Tự do - Hạnh phúc
Số lưu trữ: 24.048517
Mã Y tế: 19093496
Khoa Nhi
GIẤY RA VIỆN
- Họ tên người bệnh: PHẠM DIỆP ANH
- Ngày/tháng/năm sinh: 11/11/2018 Tuổi:6
Nam/Nữ: Nữ
- Dân tộc: Kinh  Nghề nghiệp: Học sinh
- Mã Số BHXH/ Thẻ BHYT Số: HS 4222218 73444 030019900002 22088
- Địa chỉ: 178k9, Phường Cao Thắng, Thành phố Hạ Long, Tỉnh Quảng Ninh
- Vào viện lúc: 17 giờ 13 ngày 01 tháng 12 năm 2024
Ra viện lúc: 15 giờ 30 ngày 06 tháng 12 năm 2024
- Chẩn đoán: Viêm họng cấp (J00)
- Phương pháp điều trị: kháng sinh, bù nước điện giải
- Ghi chú: Bất thường khám lại"""

    for idx, sample in enumerate([sample1, sample2], 1):
        print(f"\n{'─'*70}\nSAMPLE {idx}:")
        entities    = ner_predict(sample, tokenizer, model, id2label)
        result_json = ents_to_grv_json(entities)
        for e in entities:
            print(f"  {e['type']:25s}: {e['text']}")
        print("\nJSON:")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print("==> Format file real annotation (real_grv_annotations.json):")
    example = [
        {
            "text": (
                "Họ tên người bệnh: Nguyễn Văn An\n"
                "Năm sinh: 15/03/1975\nGiới tính: Nam\n"
                "Chẩn đoán: Tăng huyết áp (I10)"
            ),
            "entities": [
                {"start": 19, "end": 32, "label": "PATIENT_NAME"},
                {"start": 43, "end": 53, "label": "PATIENT_DOB"},
                {"start": 64, "end": 67, "label": "PATIENT_GENDER"},
                {"start": 80, "end": 104, "label": "DIAGNOSIS"},
            ],
        }
    ]
    print(json.dumps(example, ensure_ascii=False, indent=2))
    print("\n[INFO] Lưu annotations thực vào 'real_grv_annotations.json'")
    print("       để tự động merge khi train lần tiếp theo.")


if __name__ == "__main__":
    main()