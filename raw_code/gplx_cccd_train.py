"""
PhoBERT NER Pipeline cho CCCD - v2
Strategy: NER toàn bộ văn bản, tag cả LABEL lẫn VALUE
- B/I-XXX_LABEL : token thuộc phần nhãn (vd: "Họ và tên", "Date of birth")
- B/I-XXX_VALUE : token thuộc phần giá trị thực sự cần extract

Khi predict → chỉ lấy XXX_VALUE, bỏ XXX_LABEL
→ Model học ngữ cảnh tốt hơn, không bị nhầm label thành value
"""

import json
import os
import random
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# =========================
# 0. CẤU HÌNH CHUNG
# =========================

MODEL_NAME  = "vinai/phobert-base"
OUTPUT_DIR  = "phobert_cccd_ner_v2"
NUM_SAMPLES = 1500       # Tăng lên để model học đủ context
MAX_LENGTH  = 256

# Entity types cần extract (chỉ VALUE)
ENTITY_TYPES = [
    "ID_NUMBER",
    "FULL_NAME",
    "DOB",
    "GENDER",
    "NATIONALITY",
    "PLACE_OF_ORIGIN",
    "PLACE_OF_RESIDENCE",
    "DATE_OF_EXPIRY",
]

# Map entity type → field trong record
ENTITY_TO_FIELD = {
    "ID_NUMBER":          "id_cccd",
    "FULL_NAME":          "name",
    "DOB":                "dob",
    "GENDER":             "gender",
    "NATIONALITY":        "nationality",
    "PLACE_OF_ORIGIN":    "place_of_origin",
    "PLACE_OF_RESIDENCE": "address",
    "DATE_OF_EXPIRY":     "expiry_date",
}

# Tất cả tag có thể có (mỗi entity có cả _LABEL lẫn _VALUE)
ALL_TAG_PREFIXES = ["O"]
for etype in ENTITY_TYPES:
    for role in ["LABEL", "VALUE"]:
        ALL_TAG_PREFIXES.append(f"B-{etype}_{role}")
        ALL_TAG_PREFIXES.append(f"I-{etype}_{role}")


# =========================
# 1. TẠO DỮ LIỆU SYNTHETIC
# =========================

HO_LIST = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô"]
LOTEN_LIST = [
    "Văn An", "Văn Bình", "Văn Cường", "Thị Hoa", "Thị Lan",
    "Minh Anh", "Quốc Huy", "Đức Thắng", "Ngọc Hà", "Thị Mai",
    "Hữu Đức", "Thanh Tùng", "Thị Thu", "Văn Nam", "Thị Ngọc",
    "Bảo Châu", "Gia Huy", "Hoàng Long", "Thị Bích", "Văn Toàn",
]
CITY_LIST = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "An Giang", "Khánh Hòa", "Nghệ An",
    "Thanh Hóa", "Bắc Giang", "Thái Nguyên", "Quảng Nam",
]
STREET_LIST = [
    "Trần Hưng Đạo", "Nguyễn Huệ", "Lê Lợi", "Phan Chu Trinh",
    "Lý Thường Kiệt", "Hai Bà Trưng", "Đinh Tiên Hoàng", "Nguyễn Trãi",
    "Cách Mạng Tháng 8", "Nam Kỳ Khởi Nghĩa", "Điện Biên Phủ",
]
DISTRICT_LIST = [
    "Quận 1", "Quận 3", "Quận 5", "Quận Bình Thạnh", "Huyện Củ Chi",
    "Quận Hải Châu", "Quận Ngũ Hành Sơn", "Quận Hoàn Kiếm",
    "Quận Cầu Giấy", "Huyện Gia Lâm", "Quận Thanh Xuân",
]

# OCR noise patterns thực tế hay gặp
OCR_NOISE_TOKENS = [
    "1996", "1970", "1971", "1992", "1988", "2001",
    "Star", "It", "0", "AND", "OF", "THE",
    "triển", "THỊ", "Thuật",
]

# Các label text thực tế trên CCCD (đa dạng)
LABEL_TEXTS = {
    "ID_NUMBER": [
        "Số định danh cá nhân",
        "Personal identification number",
        "Số định danh cá nhân/Personal identification number",
        "Số CCCD",
        "No.",
        "Số:",
    ],
    "FULL_NAME": [
        "Họ, chữ đệm và tên khai sinh",
        "Full name",
        "Họ, chữ đệm và tên khai sinh /Full name:",
        "Họ và tên / Full name:",
        "Họ tên:",
        "Tên:",
        "Họ và tên:",
    ],
    "DOB": [
        "Ngày, tháng, năm sinh",
        "Date of birth",
        "Ngày, tháng, năm sinh /Date of birth",
        "Ngày sinh / Date of birth:",
        "Ngày sinh:",
        "Sinh ngày:",
    ],
    "GENDER": [
        "Giới tính",
        "Sex",
        "Giới tính/Sex",
        "Giới tinh/Song",   # OCR noise thực tế
        "Giới tính / Sex:",
    ],
    "NATIONALITY": [
        "Quốc tịch",
        "Nationality",
        "Quốc tịch /Nationality",
        "Quốc tịch / Nationality:",
    ],
    "PLACE_OF_ORIGIN": [
        "Quê quán",
        "Place of origin",
        "Quê quán / Place of origin:",
        "Quê quán:",
    ],
    "PLACE_OF_RESIDENCE": [
        "Nơi thường trú",
        "Place of residence",
        "Nơi thường trú / Place of residence:",
        "Nơi thường trú:",
        "Địa chỉ:",
    ],
    "DATE_OF_EXPIRY": [
        "Có giá trị đến",
        "Date of expiry",
        "Có giá trị đến / Date of expiry:",
        "Có giá trị đến:",
        "Hết hạn:",
        "Hạn đến:",
    ],
}


def random_cccd() -> str:
    return "".join([str(random.randint(0, 9)) for _ in range(12)])


def random_date(start_year=1960, end_year=2005) -> str:
    start = datetime(start_year, 1, 1)
    end   = datetime(end_year, 12, 31)
    delta = end - start
    d = start + timedelta(days=random.randint(0, delta.days))
    return d.strftime("%d/%m/%Y")


def random_issue_and_expiry() -> Tuple[str, str]:
    issue  = datetime(2016, 1, 1) + timedelta(days=random.randint(0, 3000))
    expiry = issue + timedelta(days=3650)
    return issue.strftime("%d/%m/%Y"), expiry.strftime("%d/%m/%Y")


def random_name() -> str:
    ho     = random.choice(HO_LIST)
    lo_ten = random.choice(LOTEN_LIST)
    return f"{ho} {lo_ten}".upper()


def random_address() -> str:
    num      = random.randint(1, 300)
    street   = random.choice(STREET_LIST)
    district = random.choice(DISTRICT_LIST)
    city     = random.choice(CITY_LIST)
    return f"Số {num} {street}, {district}, {city}"


def generate_records(n: int = NUM_SAMPLES) -> List[Dict]:
    records = []
    for _ in range(n):
        dob, (issue, expiry) = random_date(), random_issue_and_expiry()
        rec = {
            "id_cccd":          random_cccd(),
            "name":             random_name(),
            "dob":              random_date(),
            "gender":           random.choice(["Nam", "Nữ"]),
            "nationality":      "Việt Nam",
            "place_of_origin":  random.choice(CITY_LIST),
            "address":          random_address(),
            "date_of_issue":    issue,
            "expiry_date":      expiry,
        }
        records.append(rec)
    return records


# =========================
# 2. BUILD TEXT + SPAN MAP
# =========================

def add_ocr_noise_to_line(line: str, noise_prob: float = 0.25) -> str:
    """Chèn OCR noise token vào đầu hoặc cuối dòng."""
    if random.random() < noise_prob:
        noise = random.choice(OCR_NOISE_TOKENS)
        if random.random() < 0.5:
            return noise + " " + line
        else:
            return line + " " + noise
    return line


def record_to_text_and_spans(rec: Dict) -> Tuple[str, List[Tuple[int, int, str, str]]]:
    """
    Tạo text OCR và trả về danh sách spans:
      [(start, end, entity_type, role), ...]
    role = "LABEL" hoặc "VALUE"

    Cách hoạt động:
    - Mỗi dòng thông tin gồm: <label_text> <separator> <value_text>
    - Ta track chính xác char offset của từng phần
    """
    spans: List[Tuple[int, int, str, str]] = []

    # Header lines (không tag entity)
    header_choices = [
        ["CĂN CƯỚC CÔNG DÂN", "Citizen Identity Card"],
        ["CĂN CƯỚC", "IDENTITY CARD"],
        ["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "CĂN CƯỚC CÔNG DÂN"],
    ]
    header_lines = random.choice(header_choices)
    # Thêm noise vào header với xác suất cao hơn
    noisy_headers = [add_ocr_noise_to_line(h, noise_prob=0.4) for h in header_lines]

    # Build từng dòng thông tin
    info_configs = [
        ("ID_NUMBER",          rec["id_cccd"]),
        ("FULL_NAME",          rec["name"]),
        ("DOB",                rec["dob"]),
        ("GENDER",             rec["gender"]),
        ("NATIONALITY",        rec["nationality"]),
        ("PLACE_OF_ORIGIN",    rec["place_of_origin"]),
        ("PLACE_OF_RESIDENCE", rec["address"]),
        ("DATE_OF_EXPIRY",     rec["expiry_date"]),
    ]

    # Shuffle thứ tự các field (trừ ID thường ở đầu)
    id_config   = info_configs[0]
    rest_config = info_configs[1:]
    random.shuffle(rest_config)
    info_configs = [id_config] + rest_config

    # Separator giữa label và value
    SEPARATORS = [": ", " : ", "\n", " ", "/ ", ": \n"]

    info_lines = []
    for etype, value in info_configs:
        label_text = random.choice(LABEL_TEXTS[etype])
        sep        = random.choice(SEPARATORS)

        # GENDER + NATIONALITY đôi khi trên cùng dòng
        if etype == "GENDER" and random.random() < 0.4:
            nat_label = random.choice(LABEL_TEXTS["NATIONALITY"])
            nat_sep   = random.choice([" ", "  ", "   "])
            line = f"{label_text}{sep}{rec['gender']}{nat_sep}{nat_label}{sep}{rec['nationality']}"
            info_lines.append((line, [
                (etype,        label_text, rec["gender"]),
                ("NATIONALITY", nat_label,  rec["nationality"]),
            ]))
        else:
            line = f"{label_text}{sep}{value}"
            info_lines.append((line, [(etype, label_text, value)]))

    # Thêm OCR noise vào một số dòng
    noisy_info_lines = []
    for line, configs in info_lines:
        noisy_line = add_ocr_noise_to_line(line, noise_prob=0.3)
        noisy_info_lines.append((noisy_line, configs))

    # Ghép tất cả thành text
    all_line_texts = noisy_headers + [l for l, _ in noisy_info_lines]
    full_text = "\n".join(all_line_texts)

    # Tính spans (char offset trong full_text)
    # Ta cần tìm label_text và value trong full_text
    for _, configs in noisy_info_lines:
        for etype, label_text, value in configs:
            # Tìm label span
            label_start = full_text.find(label_text)
            if label_start != -1:
                label_end = label_start + len(label_text)
                spans.append((label_start, label_end, etype, "LABEL"))

            # Tìm value span (sau label)
            search_from  = label_start + len(label_text) if label_start != -1 else 0
            value_start  = full_text.find(value, search_from)
            if value_start != -1:
                value_end = value_start + len(value)
                spans.append((value_start, value_end, etype, "VALUE"))

    return full_text, spans


# =========================
# 3. TOKENIZE + GÁN NHÃN BIO
# =========================

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Tách token theo whitespace, trả về (token, start, end) trong text gốc.
    Xử lý đúng newline bằng cách replace thành space trước.
    """
    flat = text.replace("\n", " ")
    tokens_with_pos = []
    pos = 0
    for tok in flat.split(" "):
        if not tok:
            pos += 1
            continue
        start = flat.find(tok, pos)
        if start == -1:
            pos += len(tok) + 1
            continue
        end = start + len(tok)
        tokens_with_pos.append((tok, start, end))
        pos = end + 1
    return tokens_with_pos


def assign_bio_labels(
    tokens_with_pos: List[Tuple[str, int, int]],
    spans: List[Tuple[int, int, str, str]]
) -> List[str]:
    """
    Gán nhãn BIO cho mỗi token dựa trên char spans.
    Mỗi token chỉ thuộc 1 entity (ưu tiên VALUE hơn LABEL nếu overlap).
    """
    labels = ["O"] * len(tokens_with_pos)

    # Sort spans: VALUE trước LABEL (để VALUE được gán khi overlap)
    sorted_spans = sorted(spans, key=lambda x: (0 if x[3] == "VALUE" else 1, x[0]))

    # Với mỗi span, tìm các token bên trong và gán BIO
    for span_start, span_end, etype, role in sorted_spans:
        tag_name = f"{etype}_{role}"
        first_in_span = True

        for i, (tok, tok_start, tok_end) in enumerate(tokens_with_pos):
            # Token overlap với span?
            if tok_end <= span_start or tok_start >= span_end:
                continue

            # Chỉ gán nếu token chưa có label hoặc label hiện tại là O
            # (VALUE sẽ override LABEL nếu cùng vị trí — nhờ sort ở trên)
            if labels[i] == "O" or (role == "VALUE" and labels[i].endswith("_LABEL")):
                if first_in_span:
                    labels[i] = f"B-{tag_name}"
                    first_in_span = False
                else:
                    labels[i] = f"I-{tag_name}"

    return labels


def build_labeled_samples(records: List[Dict]) -> List[Dict]:
    """Tạo danh sách sample {tokens, ner_tags} từ records."""
    samples = []
    for rec in records:
        text, spans = record_to_text_and_spans(rec)
        tokens_with_pos = tokenize_with_offsets(text)
        if not tokens_with_pos:
            continue
        tokens = [t for t, _, _ in tokens_with_pos]
        labels = assign_bio_labels(tokens_with_pos, spans)
        samples.append({"tokens": tokens, "ner_tags": labels})
    return samples


# =========================
# 4. HuggingFace DATASET
# =========================

def samples_to_hf_dataset(samples: List[Dict]) -> Dataset:
    return Dataset.from_list([
        {"tokens": s["tokens"], "ner_tags": s["ner_tags"]}
        for s in samples
    ])


def get_label_list(samples: List[Dict]) -> List[str]:
    """Lấy toàn bộ label duy nhất từ samples, đảm bảo O ở đầu."""
    tag_set = {"O"}
    for s in samples:
        for t in s["ner_tags"]:
            tag_set.add(t)
    # O trước, còn lại sort
    return ["O"] + sorted(tag_set - {"O"})


# =========================
# 5. TOKENIZE & ALIGN LABELS
# =========================

def tokenize_and_align_labels(examples, tokenizer, label2id):
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for i, tokens in enumerate(examples["tokens"]):
        ner_tags = examples["ner_tags"][i]

        input_ids = [tokenizer.cls_token_id]
        labels    = [-100]  # CLS

        for text, tag in zip(tokens, ner_tags):
            word_tokens = tokenizer.encode(text, add_special_tokens=False)
            if not word_tokens:
                continue
            input_ids.extend(word_tokens)
            labels.append(label2id.get(tag, label2id.get("O", 0)))
            labels.extend([-100] * (len(word_tokens) - 1))  # subwords

        input_ids.append(tokenizer.sep_token_id)
        labels.append(-100)  # SEP

        # Truncate
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            labels    = labels[:MAX_LENGTH]

        padding_len = MAX_LENGTH - len(input_ids)
        mask        = [1] * len(input_ids) + [0] * padding_len
        input_ids  += [tokenizer.pad_token_id] * padding_len
        labels     += [-100] * padding_len

        all_input_ids.append(input_ids)
        all_attention_mask.append(mask)
        all_labels.append(labels)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "labels":         all_labels,
    }


# =========================
# 6. METRICS
# =========================

def compute_metrics(p, id2label):
    preds, labels = p
    preds = np.argmax(preds, axis=-1)
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
    return {"f1": f1_score(true_labels, true_preds)}


# =========================
# 7. PREDICT
# =========================

# Post-processing: validator cho VALUE entities
GENDER_NORMALIZE = {
    "NỮ": "Nữ", "NU": "Nữ", "NO": "Nữ", "N0": "Nữ",
    "NƯ": "Nữ", "NỪ": "Nữ",
    "NAM": "Nam",
}

VALIDATORS: Dict[str, callable] = {
    "ID_NUMBER":          lambda x: bool(re.match(r'^\d{9}$|^\d{12}$', x.strip())),
    "DOB":                lambda x: bool(re.match(r'^\d{2}/\d{2}/\d{4}$', x.strip())),
    "DATE_OF_EXPIRY":     lambda x: bool(re.match(r'^\d{2}/\d{2}/\d{4}$', x.strip())),
    "GENDER":             lambda x: GENDER_NORMALIZE.get(x.strip().upper(), x.strip()) in ("Nam", "Nữ"),
    "NATIONALITY":        lambda x: 1 <= len(x.split()) <= 4,
    "FULL_NAME":          lambda x: (
        bool(re.match(
            r'^[A-ZÀÁẢÃẠĂẮẶẲẴÂẤẦẨẪẬĐÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ\s]+$',
            x.strip(), re.IGNORECASE
        ))
        and 2 <= len(x.split()) <= 6
    ),
}


def ner_predict(raw_text: str, tokenizer, model, id2label: Dict) -> List[Dict]:
    """
    Predict entities từ raw_text.
    Chỉ trả về các entity có role=VALUE (bỏ LABEL).
    Áp dụng validator và normalize.
    """
    tokens_with_pos = tokenize_with_offsets(raw_text)
    if not tokens_with_pos:
        return []

    tokens = [t for t, _, _ in tokens_with_pos]

    # Build input_ids
    input_ids = [tokenizer.cls_token_id]
    # Map từ vị trí trong tokens → vị trí trong input_ids (để lấy pred)
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
        outputs = model(inputs)

    pred_ids = outputs.logits.argmax(-1)[0].tolist()

    # Map pred về token level
    token_labels = []
    for sw_start in token_to_subword_start:
        if sw_start is None or sw_start >= len(pred_ids):
            token_labels.append("O")
        else:
            token_labels.append(id2label.get(pred_ids[sw_start], "O"))

    # Decode BIO → entities (chỉ lấy VALUE)
    results = []
    current_ent: Optional[Dict] = None

    for tok, label in zip(tokens, token_labels):
        # Chỉ quan tâm _VALUE
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
        # tag_name dạng: "FULL_NAME_VALUE"
        etype = tag_name.replace("_VALUE", "")

        if prefix == "B":
            if current_ent:
                results.append(current_ent)
            current_ent = {"type": etype, "text": tok}

        elif prefix == "I":
            if current_ent and current_ent["type"] == etype:
                current_ent["text"] += " " + tok
            else:
                # I- mà không có B- trước → treat như B-
                if current_ent:
                    results.append(current_ent)
                current_ent = {"type": etype, "text": tok}

    if current_ent:
        results.append(current_ent)

    # Post-process: validate + normalize, dedup (giữ cái đầu tiên per type)
    cleaned = []
    seen_types = set()

    for ent in results:
        t    = ent["type"]
        text = ent["text"].strip()

        if t in seen_types:
            continue

        # Normalize gender
        if t == "GENDER":
            text = GENDER_NORMALIZE.get(text.upper(), text)

        # Validate
        if t in VALIDATORS and not VALIDATORS[t](text):
            print(f"[SKIP] {t}: '{text}' không hợp lệ")
            continue

        cleaned.append({"type": t, "text": text})
        seen_types.add(t)

    return cleaned


def ents_to_cccd_json(entities: List[Dict]) -> Dict:
    out = {field: "" for field in ENTITY_TO_FIELD.values()}
    for ent in entities:
        field = ENTITY_TO_FIELD.get(ent["type"])
        if field:
            out[field] = ent["text"]
    return out


# =========================
# 8. TRAIN PIPELINE
# =========================

def train_and_save(records: List[Dict]):
    print("==> Building labeled samples (label + value tagging)...")
    samples = build_labeled_samples(records)

    print(f"==> Total samples: {len(samples)}")
    random.shuffle(samples)
    n       = len(samples)
    train_s = samples[:int(0.8 * n)]
    valid_s = samples[int(0.8 * n):int(0.9 * n)]

    label_list = get_label_list(samples)
    label2id   = {l: i for i, l in enumerate(label_list)}
    id2label   = {i: l for i, l in enumerate(label_list)}

    print(f"==> Labels ({len(label_list)}): {label_list}")

    train_ds = samples_to_hf_dataset(train_s)
    valid_ds = samples_to_hf_dataset(valid_s)

    print("==> Loading tokenizer & PhoBERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    print("==> Tokenizing dataset...")
    fn_kwargs = {"tokenizer": tokenizer, "label2id": label2id}
    train_tok = train_ds.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)
    valid_tok = valid_ds.map(tokenize_and_align_labels, fn_kwargs=fn_kwargs, batched=True)

    print("==> Training...")
    args = TrainingArguments(
        OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",   # tắt fused optimizer, tương thích mọi device
    )

    def metrics_fn(p):
        return compute_metrics(p, id2label)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        processing_class=tokenizer,
        compute_metrics=metrics_fn,
    )

    trainer.train()

    print(f"==> Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("==> Done training!")

    return tokenizer, model, id2label


def load_model(output_dir: str):
    print(f"==> Loading saved model from {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(output_dir)
    id2label  = {int(k): v for k, v in model.config.id2label.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    print(f"==> Model loaded! (device: {device})")
    return tokenizer, model, id2label


# =========================
# 9. DEBUG HELPER
# =========================

def debug_sample(rec: Dict):
    """In text + span alignment để kiểm tra data generation."""
    text, spans = record_to_text_and_spans(rec)
    tokens_with_pos = tokenize_with_offsets(text)
    labels = assign_bio_labels(tokens_with_pos, spans)

    print("=" * 60)
    print("TEXT:\n", text)
    print("\nTOKEN ALIGNMENT:")
    for (tok, s, e), lab in zip(tokens_with_pos, labels):
        if lab != "O":
            print(f"  [{lab:40s}] '{tok}'")
    print("=" * 60)


# =========================
# 10. MAIN
# =========================

def main():
    if os.path.exists(OUTPUT_DIR) and os.path.isfile(os.path.join(OUTPUT_DIR, "config.json")):
        tokenizer, model, id2label = load_model(OUTPUT_DIR)
    else:
        print("==> No saved model found. Starting training pipeline...")
        records = generate_records(NUM_SAMPLES)

        # Debug 1 sample trước khi train
        print("\n[DEBUG] Kiểm tra data generation:")
        debug_sample(records[0])

        tokenizer, model, id2label = train_and_save(records)

    # =========================================================
    # PREDICT với text OCR thực tế (nhiều noise)
    # =========================================================
    print("\n" + "=" * 60)
    print("==> Demo predict với OCR text thực tế...")

    sample_text = """CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc SOCIALIST REPUBLIC OF VIET NAM Independence - Freedom - Happiness

CĂN CƯỚC CÔNG DÂN Citizen Identity Card

Số / No.: 079203030140 Họ và tên / Full name: NGUYỄN MINH QUÂN Ngày sinh / Date of birth: 14/04/2003 Giới tính / Sex: Nam Quốc tịch / Nationality: Việt Nam Quê quán / Place of origin: P. Tân Phú, Quận 7, TP. Hồ Chí Minh Nơi thường trú / Place of residence: 803/23/10/13 H. Tấn Phát, KP2, P. Phú Thuận, Q.7, TP. HCM"""

    print("\nRAW TEXT:\n", sample_text)
    print("\n" + "-" * 60)

    entities = ner_predict(sample_text, tokenizer, model, id2label)
    print("\nENTITIES (chỉ VALUE):")
    for e in entities:
        print(f"  {e['type']:25s}: {e['text']}")

    result_json = ents_to_cccd_json(entities)
    print("\nJSON OUTPUT:")
    print(json.dumps(result_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()