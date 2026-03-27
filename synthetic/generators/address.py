"""Address generator utility for Vietnamese addresses."""

import random
from typing import List


PROVINCE_LIST: List[str] = [
    "Hà Nội", "Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ",
    "Bình Dương", "Đồng Nai", "An Giang", "Khánh Hòa", "Nghệ An",
    "Thanh Hóa", "Lâm Đồng", "Quảng Nam", "Quảng Ninh", "Huế",
    "Bắc Giang", "Thái Nguyên", "Gia Lai", "Đắk Lắk", "Kiên Giang",
]

DISTRICT_LIST: List[str] = [
    "Quận 1", "Quận 3", "Quận 5", "Quận 7", "Quận Bình Thạnh",
    "Huyện Củ Chi", "Huyện Lâm Hà", "Huyện Quế Sơn",
    "Thành phố Hạ Long", "Quận Hoàn Kiếm", "Huyện Gia Lâm",
    "Huyện Đức Trọng", "Quận Hải Châu", "Quận Ngũ Hành Sơn",
    "Quận Cầu Giấy", "Quận Thanh Xuân",
]

WARD_LIST: List[str] = [
    "Phường Tân Phú", "Xã Tân Hà", "Xã Quế Xuân 2", "Phường Cao Thắng",
    "Thôn Phủ Mỹ", "Phường Lộc Sơn", "Phường 3", "Xã Lộc Tân",
    "Phường Phú Thuận",
]

STREET_LIST: List[str] = [
    "Trần Hưng Đạo", "Nguyễn Huệ", "Lê Lợi", "Phan Chu Trinh",
    "Hai Bà Trưng", "Đinh Tiên Hoàng", "Cách Mạng Tháng 8",
    "Nam Kỳ Khởi Nghĩa", "H. Tấn Phát", "Lý Thường Kiệt",
    "Nguyễn Trãi", "Điện Biên Phủ",
]

HAMLET_NAMES: List[str] = ["Phủ Mỹ", "Trung", "An", "Tân"]


class AddressGenerator:
    """Utility for generating random Vietnamese addresses."""

    def random_province(self) -> str:
        return random.choice(PROVINCE_LIST)

    def random_district(self) -> str:
        return random.choice(DISTRICT_LIST)

    def random_ward(self) -> str:
        return random.choice(WARD_LIST)

    def random_street(self) -> str:
        return random.choice(STREET_LIST)

    def generate(self) -> str:
        """Generate a random Vietnamese address."""
        styles = [
            self._hamlet_style,
            self._street_number_style,
            self._ward_only_style,
            self._alley_style,
        ]
        return random.choice(styles)()

    def generate_city_only(self) -> str:
        """Generate just a province/city name."""
        return random.choice(PROVINCE_LIST)

    def _hamlet_style(self) -> str:
        hamlet = random.choice(HAMLET_NAMES)
        return (
            f"Thôn {hamlet}, {random.choice(WARD_LIST)}, "
            f"{random.choice(DISTRICT_LIST)}, {random.choice(PROVINCE_LIST)}"
        )

    def _street_number_style(self) -> str:
        num = random.randint(1, 300)
        return (
            f"Số {num} {random.choice(STREET_LIST)}, "
            f"{random.choice(WARD_LIST)}, {random.choice(DISTRICT_LIST)}, "
            f"{random.choice(PROVINCE_LIST)}"
        )

    def _ward_only_style(self) -> str:
        return (
            f"{random.choice(WARD_LIST)}, {random.choice(DISTRICT_LIST)}, "
            f"{random.choice(PROVINCE_LIST)}"
        )

    def _alley_style(self) -> str:
        a = random.randint(1, 999)
        b = random.randint(1, 99)
        c = random.randint(1, 30)
        kp = random.randint(1, 5)
        return (
            f"{a}/{b}/{c} {random.choice(STREET_LIST)}, KP{kp}, "
            f"{random.choice(WARD_LIST)}, {random.choice(PROVINCE_LIST)}"
        )
