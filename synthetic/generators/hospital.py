"""Hospital generator for Giấy ra viện (discharge papers) documents."""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from synthetic.generators.base import BaseGenerator
from synthetic.generators.address import AddressGenerator


HO_LIST: List[str] = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý",
]

LOTEN_LIST: List[str] = [
    "Văn An", "Văn Bình", "Thị Hoa", "Thị Lan", "Minh Anh", "Quốc Huy",
    "Đức Thắng", "Ngọc Hà", "Thị Mai", "Hữu Đức", "Thanh Tùng", "Thị Thu",
    "Văn Nam", "Thị Ngọc", "Bảo Châu", "Gia Huy", "Hoàng Long", "Thị Bích",
    "Văn Toàn", "Thị Hằng", "Đình Phúc", "Thị Tuyết", "Xuân Trường",
    "Diệp Anh", "Thị Hà",
]

ETHNICITY_LIST: List[str] = [
    "Kinh", "Tày", "Thái", "Mường", "Khmer", "Hoa", "Nùng",
]

OCCUPATION_LIST: List[str] = [
    "Nông dân", "Công nhân", "Giáo viên", "Bác sĩ", "Kỹ sư", "Buôn bán",
    "Học sinh", "Sinh viên", "Hưu trí", "Nội trợ", "Người già",
    "Lao động tự do",
]

HOSPITAL_LIST: List[Tuple[str, str]] = [
    ("Bệnh viện Đa khoa tỉnh Lâm Đồng", "Sở Y tế Lâm Đồng"),
    ("Bệnh viện Đa khoa Vĩnh Đức", "Sở Y tế Quảng Nam"),
    ("Bệnh viện Đa khoa tỉnh Quảng Ninh", "Sở Y tế Quảng Ninh"),
    ("Bệnh viện Đa khoa tỉnh Nghệ An", "Sở Y tế Nghệ An"),
    ("Bệnh viện Đa khoa Trung ương Huế", "Bộ Y tế"),
    ("Bệnh viện Nhân dân 115", "Sở Y tế TP. Hồ Chí Minh"),
    ("Bệnh viện Bạch Mai", "Bộ Y tế"),
    ("Bệnh viện Chợ Rẫy", "Bộ Y tế"),
    ("Bệnh viện Đa khoa Gia Lai", "Sở Y tế Gia Lai"),
    ("Bệnh viện Đa khoa tỉnh Bình Dương", "Sở Y tế Bình Dương"),
]

DEPARTMENT_LIST: List[str] = [
    "Khoa Nội A", "Khoa Nội Tổng hợp",
    "Khoa Ngoại Chấn thương Chỉnh hình", "Khoa Nhi",
    "Khoa Hồi sức tích cực", "Khoa Tim mạch", "Khoa Thần kinh",
    "Khoa Sản", "Khoa Tai Mũi Họng", "Khoa Mắt", "Khoa Da liễu",
    "Khoa Tiêu hóa", "Khoa Ung bướu", "Khoa Cấp cứu", "Khoa Nội thận",
]

DIAGNOSIS_LIST: List[Tuple[str, str]] = [
    ("I63", "Nhồi máu não"),
    ("J00", "Viêm họng cấp"),
    ("M51.1", "Bệnh đĩa đệm cột sống thắt lưng có tổn thương rễ tủy"),
    ("I10", "Tăng huyết áp nguyên phát"),
    ("E11", "Đái tháo đường type 2"),
    ("J18.9", "Viêm phổi không xác định"),
    ("K29.5", "Viêm dạ dày mạn tính"),
    ("N18", "Bệnh thận mạn"),
    ("S82.0", "Gãy xương bánh chè"),
    ("C34", "Ung thư phế quản và phổi"),
    ("G40", "Động kinh"),
    ("A91", "Sốt xuất huyết Dengue"),
    ("I50", "Suy tim"),
    ("B34.9", "Nhiễm virus không xác định"),
]

TREATMENT_LIST: List[str] = [
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

NOTE_LIST: List[str] = [
    "Tái khám sau 7 ngày",
    "Khám lại khi có bất thường",
    "Uống thuốc theo đơn, thay băng hằng ngày tại trạm y tế cơ sở",
    "Cắt chỉ sau 10 ngày, tái khám định kỳ",
    "Tiếp tục điều trị ngoại trú theo đơn",
    "Nghỉ ngơi tại nhà, hạn chế vận động nặng",
    "Khám lại tại phòng khám ngoại trú, mang theo kết quả xét nghiệm",
]


class HospitalGenerator(BaseGenerator):
    """Generator for Giấy ra viện document fields."""

    def __init__(self) -> None:
        self._address_gen = AddressGenerator()

    def generate(self) -> Dict[str, str]:
        """Generate a hospital discharge record."""
        hosp_name, dept_mgmt = random.choice(HOSPITAL_LIST)
        diag_code, diag_name = random.choice(DIAGNOSIS_LIST)

        if random.random() < 0.4:
            extra_code, extra_name = random.choice(DIAGNOSIS_LIST)
            diagnosis = f"{diag_name} ({diag_code}); {extra_name} ({extra_code})"
        else:
            diagnosis = f"{diag_name} ({diag_code})"

        admission, discharge = self._random_admission_discharge()

        return {
            "hospital_name": hosp_name,
            "dept_mgmt": dept_mgmt,
            "department": random.choice(DEPARTMENT_LIST),
            "medical_code": self._random_medical_code(),
            "patient_name": self._random_name(),
            "patient_dob": self._random_date(),
            "patient_gender": random.choice(["Nam", "Nữ"]),
            "patient_ethnicity": random.choice(ETHNICITY_LIST),
            "patient_occupation": random.choice(OCCUPATION_LIST),
            "patient_address": self._address_gen.generate(),
            "bhxh_code": self._random_bhxh_code(),
            "admission_date": admission,
            "discharge_date": discharge,
            "diagnosis": diagnosis,
            "treatment_method": random.choice(TREATMENT_LIST),
            "notes": random.choice(NOTE_LIST),
        }

    def _random_name(self) -> str:
        """Generate a Vietnamese full name with random casing."""
        ho = random.choice(HO_LIST)
        lo_ten = random.choice(LOTEN_LIST)
        full = f"{ho} {lo_ten}"
        style = random.random()
        if style < 0.4:
            return full.upper()
        elif style < 0.7:
            return full
        return full.lower()

    def _random_date(self, start_year: int = 1940, end_year: int = 2020) -> str:
        """Generate a random date, sometimes year-only."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        if random.random() < 0.3:
            return d.strftime("%Y")
        return d.strftime("%d/%m/%Y")

    def _random_admission_discharge(self) -> Tuple[str, str]:
        """Generate admission and discharge datetime strings."""
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
                d=str(dt.day),
                mo=str(dt.month),
                y=str(dt.year),
            )

        return _fmt(admit, fmt_admit), _fmt(disch, fmt_disch)

    def _random_bhxh_code(self) -> str:
        """Generate a BHXH/BHYT insurance code."""
        styles = [
            lambda: "GD" + "".join(
                [str(random.randint(0, 9)) for _ in range(13)]
            ),
            lambda: "HS " + " ".join([
                "".join([str(random.randint(0, 9))
                         for _ in range(random.randint(3, 5))])
                for _ in range(3)
            ]),
            lambda: (
                "GD " + str(random.randint(1, 9)) + " "
                + str(random.randint(10, 99)) + " "
                + "".join([str(random.randint(0, 9)) for _ in range(11)])
            ),
        ]
        return random.choice(styles)()

    def _random_medical_code(self) -> str:
        """Generate a medical record code."""
        year = random.randint(20, 24)
        seq = random.randint(1000, 99999)
        return f"{year}.{seq:06d}"
