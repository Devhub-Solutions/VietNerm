from vncv.ocr import extract_text
from vietnerm import VietNerm


text = extract_text(r"C:\Users\ongdevtre\Pictures\Screenshots\Screenshot 2026-03-28 125541.png")
rawtext = "\n".join(text)
print("--- OCR RAW TEXT ---")
print(rawtext)
print("--------------------")

ner = VietNerm(doc_type="gplx", hf_username="ngocthanhdoan")
result = ner.extract(rawtext)
for key, value in result.items():
    print(f"{key}: {value}")