"""Giấy khai sinh (Birth Certificate) generator."""

import random
from datetime import datetime, timedelta
from typing import Dict, List

from synthetic.generators.base import BaseGenerator
from synthetic.generators.address import AddressGenerator


HO_LIST: List[str] = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Phan", "Vũ", "Võ",
    "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý",
]

LOTEN_LIST: List[str] = [
    "Minh Khôi", "Gia Bảo", "Phúc Lâm", "Bảo Lâm", "Bảo Trâm", "Hải Đăng",
    "Thanh Thúy", "Nhật Linh", "Gia Khiêm", "Minh Thắng", "Gia Hân",
    "Thiên An", "Tú Anh", "Hạ Linh", "Mạnh Cường", "Thị Trang",
]


class KhaiSinhGenerator(BaseGenerator):
    """Generator for Giấy khai sinh document fields."""

    def __init__(self) -> None:
        self._address_gen = AddressGenerator()

    def generate(self) -> Dict[str, str]:
        """Generate a Giấy khai sinh record."""
        dob = datetime(2010, 1, 1) + timedelta(days=random.randint(0, 5000))
        reg_date = dob + timedelta(days=random.randint(1, 30))
        
        return {
            "id_number": "".join([str(random.randint(0, 9)) for _ in range(12)]),
            "full_name": self._random_name().upper(),
            "gender": random.choice(["Nam", "Nữ"]),
            "date_of_birth": dob.strftime("%d/%m/%Y"),
            "dob_text": "Một ngày nào đó",  # Basic placeholder for dob_text
            "place_of_birth": self._address_gen.generate_city_only(),
            "ethnicity": "Kinh",
            "nationality": "Việt Nam",
            "father_name": self._random_name().upper(),
            "mother_name": self._random_name().upper(),
            "registration_place": self._address_gen.generate_city_only(),
            "registration_date": reg_date.strftime("%d/%m/%Y"),
        }

    def _random_name(self) -> str:
        """Generate a random Vietnamese name."""
        ho = random.choice(HO_LIST)
        lo_ten = random.choice(LOTEN_LIST)
        return f"{ho} {lo_ten}"
