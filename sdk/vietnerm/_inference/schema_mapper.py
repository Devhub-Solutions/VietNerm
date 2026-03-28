"""
Schema Mapper - Map BIO predictions back to structured entity dictionaries.

Loads schema.yaml for a given document type to know expected fields,
then maps NER predictions to a structured output dict.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .postprocess import clean_entity_boundaries, filter_validated_entities


class SchemaMapper:
    """Map raw NER entity predictions to structured document fields.

    Args:
        doc_type: Document type identifier (e.g., 'cccd', 'giay_ra_vien').
        schema_path: Path to schema.yaml file. If None, auto-resolved from
            templates/{doc_type}/schema.yaml.
        entity_to_field: Optional explicit mapping from entity type to output field.
    """

    # Default entity-to-field mappings per document type.
    # Keys are UPPERCASE entity types (as produced by _normalize_etype in postprocess.py).
    # Supports both label schemes:
    #   HF Hub: B-id_number      -> ID_NUMBER
    #   Local:  B-ID_NUMBER_VALUE -> ID_NUMBER
    DEFAULT_MAPPINGS: Dict[str, Dict[str, str]] = {
        "cccd": {
            "ID_NUMBER": "id_number",
            "FULL_NAME": "name",
            "DATE_OF_BIRTH": "date_of_birth",  # HF Hub scheme
            "DOB": "date_of_birth",             # old local scheme alias
            "GENDER": "gender",
            "NATIONALITY": "nationality",
            "PLACE_OF_ORIGIN": "place_of_origin",
            "PLACE_OF_RESIDENCE": "address",
            "DATE_OF_EXPIRY": "expiry_date",
        },
        "giay_ra_vien": {
            "HOSPITAL_NAME": "hospital_name",
            "DEPARTMENT": "department",
            "MEDICAL_CODE": "medical_code",
            "PATIENT_NAME": "patient_name",
            "PATIENT_DOB": "patient_dob",
            "PATIENT_GENDER": "patient_gender",
            "PATIENT_ETHNICITY": "patient_ethnicity",
            "PATIENT_OCCUPATION": "patient_occupation",
            "PATIENT_ADDRESS": "patient_address",
            "BHXH_CODE": "bhxh_code",
            "ADMISSION_DATE": "admission_date",
            "DISCHARGE_DATE": "discharge_date",
            "DIAGNOSIS": "diagnosis",
            "TREATMENT_METHOD": "treatment_method",
            "NOTES": "notes",
        },
        "vehicle_registration": {
            "PLATE_NUMBER": "plate_number",
            "OWNER_NAME": "owner_name",
            "OWNER_ADDRESS": "owner_address",
            "BRAND": "brand",
            "VEHICLE_TYPE": "vehicle_type",
            "ENGINE_NUMBER": "engine_number",
            "CHASSIS_NUMBER": "chassis_number",
            "MANUFACTURE_YEAR": "manufacture_year",
        },
    }

    def __init__(
        self,
        doc_type: str,
        schema_path: Optional[str] = None,
        entity_to_field: Optional[Dict[str, str]] = None,
    ) -> None:
        self.doc_type = doc_type
        self.schema: Optional[Dict[str, Any]] = None
        self.expected_fields: List[str] = []

        # Load schema if available
        if schema_path:
            schema_file = Path(schema_path)
        else:
            schema_file = Path("templates") / doc_type / "schema.yaml"

        if schema_file.exists():
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schema = yaml.safe_load(f)
            if self.schema and "entities" in self.schema:
                self.expected_fields = [
                    ent["name"] for ent in self.schema["entities"]
                ]

        # Entity type → output field mapping
        if entity_to_field:
            self.entity_to_field = entity_to_field
        elif doc_type in self.DEFAULT_MAPPINGS:
            self.entity_to_field = self.DEFAULT_MAPPINGS[doc_type]
        elif self.schema and "entities" in self.schema:
            # Dynamically build mapping from schema entities
            # Label scheme convention: name 'full_name' -> entity type 'FULL_NAME'
            self.entity_to_field = {
                ent["name"].upper(): ent["name"]
                for ent in self.schema["entities"]
                if "name" in ent
            }
        else:
            self.entity_to_field = {}

    def map_entities(
        self,
        raw_entities: List[Dict],
        validate: bool = True,
        clean: bool = True,
    ) -> Dict[str, str]:
        """Map raw NER entity predictions to a structured dict.

        Args:
            raw_entities: List of entity dicts from NERPipeline.predict().
            validate: Whether to apply entity-type validators.
            clean: Whether to clean/normalize entity text.

        Returns:
            Dict mapping field names to extracted values.
            Example: {"name": "Nguyễn Văn A", "date_of_birth": "01/01/1990", ...}
        """
        entities = raw_entities

        if clean:
            entities = clean_entity_boundaries(entities)

        if validate:
            entities = filter_validated_entities(entities)

        # Build output dict with all expected fields initialized to empty
        output: Dict[str, str] = {}
        if self.expected_fields:
            output = {field: "" for field in self.expected_fields}
        else:
            output = {field: "" for field in self.entity_to_field.values()}

        for ent in entities:
            field = self.entity_to_field.get(ent["type"])
            if field and field in output:
                output[field] = ent["text"]

        return output

    def map_entities_with_confidence(
        self,
        raw_entities: List[Dict],
        validate: bool = True,
        clean: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Map entities and include confidence scores.

        Args:
            raw_entities: List of entity dicts from NERPipeline.predict().
            validate: Whether to apply entity-type validators.
            clean: Whether to clean/normalize entity text.

        Returns:
            Dict mapping field names to {"value": str, "confidence": float}.
        """
        entities = raw_entities

        if clean:
            entities = clean_entity_boundaries(entities)

        if validate:
            entities = filter_validated_entities(entities)

        # Build output
        output: Dict[str, Dict[str, Any]] = {}
        all_fields = (
            self.expected_fields
            if self.expected_fields
            else list(self.entity_to_field.values())
        )
        for field in all_fields:
            output[field] = {"value": "", "confidence": 0.0}

        for ent in entities:
            field = self.entity_to_field.get(ent["type"])
            if field and field in output:
                output[field] = {
                    "value": ent["text"],
                    "confidence": ent.get("confidence", 0.0),
                }

        return output
