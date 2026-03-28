"""Synthetic data generators for Vietnamese documents."""

from synthetic.generators.base import BaseGenerator
from synthetic.generators.person import PersonGenerator
from synthetic.generators.hospital import HospitalGenerator
from synthetic.generators.vehicle import VehicleGenerator
from synthetic.generators.address import AddressGenerator
from synthetic.generators.gplx import GPLXGenerator
from synthetic.generators.giay_khai_sinh import KhaiSinhGenerator

GENERATOR_MAP = {
    "person": PersonGenerator,
    "hospital": HospitalGenerator,
    "vehicle": VehicleGenerator,
    "gplx": GPLXGenerator,
    "giay_khai_sinh": KhaiSinhGenerator,
}

__all__ = [
    "BaseGenerator",
    "PersonGenerator",
    "HospitalGenerator",
    "VehicleGenerator",
    "GPLXGenerator",
    "KhaiSinhGenerator",
    "AddressGenerator",
    "GENERATOR_MAP",
]
