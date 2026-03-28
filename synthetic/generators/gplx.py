"""GPLX (Giấy phép lái xe) generator."""

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
    "Văn An", "Văn Bình", "Thị Hoa", "Thị Lan", "Minh Anh", "Quốc Huy",
    "Đức Thắng", "Ngọc Hà", "Thị Mai", "Hữu Đức", "Thanh Tùng",
    "Văn Nam", "Thị Ngọc", "Bảo Châu", "Gia Huy", "Hoàng Long",
]

CLASS_LIST: List[str] = ["A1", "A2", "A3", "A4", "B1", "B2", "C", "D", "E"]


class GPLXGenerator(BaseGenerator):
    """Generator for GPLX document fields."""

    def __init__(self) -> None:
        self._address_gen = AddressGenerator()

    def generate(self) -> Dict[str, str]:
        """Generate a GPLX record."""
        issue = datetime(2015, 1, 1) + timedelta(days=random.randint(0, 3000))
        expiry = issue + timedelta(days=3650)
        
        return {
            "id_number": "".join([str(random.randint(0, 9)) for _ in range(12)]),
            "full_name": self._random_name(),
            "date_of_birth": self._random_date(1970, 2005),
            "nationality": "Việt Nam",
            "address": self._address_gen.generate(),
            "issue_date": issue.strftime("%d/%m/%Y"),
            "expiry_date": expiry.strftime("%d/%m/%Y"),
            "class": random.choice(CLASS_LIST),
        }

    def _random_name(self) -> str:
        """Generate a Vietnamese full name in uppercase."""
        ho = random.choice(HO_LIST)
        lo_ten = random.choice(LOTEN_LIST)
        return f"{ho} {lo_ten}".upper()

    def _random_date(self, start_year: int = 1970, end_year: int = 2005) -> str:
        """Generate a random date string in DD/MM/YYYY format."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        return d.strftime("%d/%m/%Y")
