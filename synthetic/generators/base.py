"""Base class for synthetic data generators."""

from abc import ABC, abstractmethod
from typing import Dict


class BaseGenerator(ABC):
    """Abstract base class for document field generators.

    Each generator produces a dictionary of field values
    matching the schema of a specific document type.
    """

    @abstractmethod
    def generate(self) -> Dict[str, str]:
        """Generate a single record with all fields populated.

        Returns:
            Dictionary mapping field names to generated string values.
        """
        ...

    def generate_batch(self, n: int) -> list:
        """Generate multiple records.

        Args:
            n: Number of records to generate.

        Returns:
            List of generated record dictionaries.
        """
        return [self.generate() for _ in range(n)]
