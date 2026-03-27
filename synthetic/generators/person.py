"""Person generator for CCCD (Căn cước công dân) documents."""

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
    "Văn An", "Văn Bình", "Văn Cường", "Thị Hoa", "Thị Lan",
    "Minh Anh", "Quốc Huy", "Đức Thắng", "Ngọc Hà", "Thị Mai",
    "Hữu Đức", "Thanh Tùng", "Thị Thu", "Văn Nam", "Thị Ngọc",
    "Bảo Châu", "Gia Huy", "Hoàng Long", "Thị Bích", "Văn Toàn",
    "Thị Hằng", "Đình Phúc", "Thị Tuyết", "Xuân Trường",
    "Diệp Anh", "Thị Hà",
]


class PersonGenerator(BaseGenerator):
    """Generator for CCCD document fields."""

    def __init__(self) -> None:
        self._address_gen = AddressGenerator()

    def generate(self) -> Dict[str, str]:
        """Generate a CCCD record with all entity fields."""
        issue, expiry = self._random_issue_and_expiry()
        return {
            "id_number": self._random_cccd_number(),
            "full_name": self._random_name(),
            "date_of_birth": self._random_date(1960, 2005),
            "gender": random.choice(["Nam", "Nữ"]),
            "nationality": "Việt Nam",
            "place_of_origin": self._address_gen.generate_city_only(),
            "place_of_residence": self._address_gen.generate(),
            "date_of_expiry": expiry,
        }

    def _random_cccd_number(self) -> str:
        """Generate a 12-digit CCCD number."""
        return "".join([str(random.randint(0, 9)) for _ in range(12)])

    def _random_name(self) -> str:
        """Generate a Vietnamese full name in uppercase."""
        ho = random.choice(HO_LIST)
        lo_ten = random.choice(LOTEN_LIST)
        return f"{ho} {lo_ten}".upper()

    def _random_date(self, start_year: int = 1960, end_year: int = 2005) -> str:
        """Generate a random date string in DD/MM/YYYY format."""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        return d.strftime("%d/%m/%Y")

    def _random_issue_and_expiry(self) -> tuple:
        """Generate issue and expiry date pair."""
        issue = datetime(2016, 1, 1) + timedelta(days=random.randint(0, 3000))
        expiry = issue + timedelta(days=3650)
        return issue.strftime("%d/%m/%Y"), expiry.strftime("%d/%m/%Y")
