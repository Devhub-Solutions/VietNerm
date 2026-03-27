"""Vehicle generator for vehicle registration documents."""

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

VEHICLE_TYPES: List[str] = [
    "Xe con", "Xe tải nhẹ", "Xe tải nặng", "Xe khách",
    "Xe máy", "Xe mô tô", "Xe bán tải", "Xe đầu kéo",
]

BRANDS: List[str] = [
    "Toyota", "Honda", "Hyundai", "Kia", "Mazda", "Ford",
    "VinFast", "Mitsubishi", "Suzuki", "Mercedes-Benz",
    "BMW", "Yamaha", "Piaggio", "SYM", "Thaco",
]

MODEL_CODES: List[str] = [
    "Vios 1.5E", "City RS", "Accent 1.4AT", "Morning S",
    "CX-5 2.0L", "Ranger XLS", "Lux A2.0", "Xpander AT",
    "XL7", "C200", "Wave Alpha", "Air Blade 150",
    "Liberty S", "Attila", "K250",
]

COLORS: List[str] = [
    "Trắng", "Đen", "Bạc", "Xám", "Đỏ", "Xanh dương",
    "Xanh lá", "Nâu", "Vàng", "Cam",
]

PROVINCE_CODES: List[str] = [
    "29", "30", "31", "32", "33", "34", "36", "37", "38",
    "43", "47", "48", "49", "50", "51", "59", "60", "61",
    "62", "63", "65", "66", "67", "68", "70", "71", "72",
    "73", "74", "75", "76", "77", "79", "80", "81", "82",
    "83", "84", "85", "86", "88", "89", "90", "92", "93",
    "94", "95", "97", "98", "99",
]


class VehicleGenerator(BaseGenerator):
    """Generator for vehicle registration document fields."""

    def __init__(self) -> None:
        self._address_gen = AddressGenerator()

    def generate(self) -> Dict[str, str]:
        """Generate a vehicle registration record."""
        return {
            "owner_name": self._random_name(),
            "owner_address": self._address_gen.generate(),
            "plate_number": self._random_plate(),
            "vehicle_type": random.choice(VEHICLE_TYPES),
            "brand": random.choice(BRANDS),
            "model_code": random.choice(MODEL_CODES),
            "engine_number": self._random_engine_number(),
            "chassis_number": self._random_chassis_number(),
            "color": random.choice(COLORS),
            "registration_date": self._random_date(),
        }

    def _random_name(self) -> str:
        """Generate a Vietnamese full name in uppercase."""
        ho = random.choice(HO_LIST)
        lo_ten = random.choice(LOTEN_LIST)
        return f"{ho} {lo_ten}".upper()

    def _random_plate(self) -> str:
        """Generate a Vietnamese license plate number."""
        province = random.choice(PROVINCE_CODES)
        letter = random.choice("ABCDEFGHKLMNPRSTUVXYZ")
        if random.random() < 0.5:
            # Motorcycle format: XX-YZ NNNNN
            num = random.randint(10000, 99999)
            letter2 = random.choice("0123456789")
            return f"{province}{letter}{letter2} {num}"
        else:
            # Car format: XX-Y NNN.NN
            num1 = random.randint(100, 999)
            num2 = random.randint(10, 99)
            return f"{province}{letter}-{num1}.{num2}"

    def _random_engine_number(self) -> str:
        """Generate an engine number."""
        prefix = random.choice(["E", "G", "JA", "KF", "2NR", "1NZ", "4A"])
        digits = "".join([str(random.randint(0, 9)) for _ in range(7)])
        return f"{prefix}{digits}"

    def _random_chassis_number(self) -> str:
        """Generate a chassis (frame) number."""
        prefix = random.choice([
            "RLHHA", "MHFHB", "KNAB", "MALA", "JMZGH",
            "MNBJ", "RLMTD", "MHKAA",
        ])
        digits = "".join([str(random.randint(0, 9)) for _ in range(10)])
        return f"{prefix}{digits}"

    def _random_date(self) -> str:
        """Generate a random registration date."""
        start = datetime(2015, 1, 1)
        end = datetime(2025, 12, 31)
        d = start + timedelta(days=random.randint(0, (end - start).days))
        return d.strftime("%d/%m/%Y")
