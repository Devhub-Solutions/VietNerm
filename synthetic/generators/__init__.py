"""Synthetic data generators for Vietnamese documents."""

from synthetic.generators.base import BaseGenerator
from synthetic.generators.person import PersonGenerator
from synthetic.generators.hospital import HospitalGenerator
from synthetic.generators.vehicle import VehicleGenerator
from synthetic.generators.address import AddressGenerator

GENERATOR_MAP = {
    "person": PersonGenerator,
    "hospital": HospitalGenerator,
    "vehicle": VehicleGenerator,
}

__all__ = [
    "BaseGenerator",
    "PersonGenerator",
    "HospitalGenerator",
    "VehicleGenerator",
    "AddressGenerator",
    "GENERATOR_MAP",
]
