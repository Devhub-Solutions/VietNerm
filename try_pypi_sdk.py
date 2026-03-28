from vietnerm import VietNerm

ner = VietNerm(doc_type="cccd", hf_username="ngocthanhdoan")
result = ner.extract("""
CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM
Độc lập - Tự do - Hạnh phúc

GIẤY CHỨNG NHẬN QUỐC TỊCH
Số: 001000000001

tên: NGUYỄN VĂN A
Ngày sinh: 01/01/1990
Giới tính: Nam
Nơi sinh: Hà Nội
Quốc tịch: Việt Nam
Nơi thường trú: 123 ABC, Hà Nội

Ngày cấp: 01/01/2020
Cơ quan cấp: Bộ Tư pháp

Ký tên và đóng dấu
""")
for key, value in result.items():
    print(f"{key}: {value}")