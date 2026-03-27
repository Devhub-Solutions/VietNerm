"""Template engine that renders Jinja2 templates and tracks entity positions."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Environment, BaseLoader


class _TrackingRenderer:
    """Renders a Jinja2 template while recording exact character positions of values."""

    # Unique sentinel that won't appear in real Vietnamese text
    _SENTINEL_PREFIX = "\x00ENTITY_"
    _SENTINEL_SUFFIX = "\x00"

    def __init__(self, template_text: str, schema_entities: List[str]) -> None:
        self._template_text = template_text
        self._schema_entities = set(schema_entities)

    def render(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Render template with data and track entity positions.

        Args:
            data: Dictionary mapping placeholder names to generated values.

        Returns:
            Dictionary with 'text' (rendered string) and 'entities' list.
            Each entity has: label, start, end, value.
        """
        # Wrap each entity value with sentinels for position tracking
        wrapped_data: Dict[str, str] = {}
        for key, value in data.items():
            if key in self._schema_entities:
                wrapped_data[key] = (
                    f"{self._SENTINEL_PREFIX}{key}:{value}"
                    f"{self._SENTINEL_SUFFIX}"
                )
            else:
                wrapped_data[key] = value

        env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
        template = env.from_string(self._template_text)
        rendered_with_sentinels = template.render(**wrapped_data)

        # Extract entity positions from sentinel markers
        entities: List[Dict[str, Any]] = []
        clean_text = rendered_with_sentinels
        offset_adjustment = 0

        pattern = re.compile(
            re.escape(self._SENTINEL_PREFIX)
            + r"([^:]+):(.+?)"
            + re.escape(self._SENTINEL_SUFFIX),
            re.DOTALL,
        )

        for match in pattern.finditer(rendered_with_sentinels):
            label = match.group(1)
            value = match.group(2)
            # Position in the original rendered_with_sentinels
            full_match_start = match.start()
            full_match_text = match.group(0)

            entities.append({
                "label": label,
                "value": value,
                "original_start": full_match_start,
                "original_length": len(full_match_text),
                "value_length": len(value),
            })

        # Now rebuild clean text by removing sentinels and computing positions
        clean_text = ""
        last_end = 0
        final_entities: List[Dict[str, Any]] = []

        for match in pattern.finditer(rendered_with_sentinels):
            label = match.group(1)
            value = match.group(2)

            # Add text before this match
            clean_text += rendered_with_sentinels[last_end:match.start()]
            # Record entity start position in clean text
            entity_start = len(clean_text)
            clean_text += value
            entity_end = len(clean_text)

            final_entities.append({
                "label": label,
                "start": entity_start,
                "end": entity_end,
                "value": value,
            })

            last_end = match.end()

        # Add remaining text
        clean_text += rendered_with_sentinels[last_end:]

        return {
            "text": clean_text,
            "entities": final_entities,
        }


class TemplateEngine:
    """Loads and renders Jinja2 templates with entity position tracking.

    Reads templates from the filesystem, renders them with generated data,
    and returns structured output with exact character positions for each entity.
    """

    def __init__(self, templates_dir: Path, schema_path: Path) -> None:
        """Initialize the template engine.

        Args:
            templates_dir: Path to directory containing template_*.txt files.
            schema_path: Path to schema.yaml defining entity fields.
        """
        self._templates_dir = Path(templates_dir)
        self._schema_path = Path(schema_path)
        self._templates: List[str] = []
        self._schema_entities: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load schema and template files."""
        # Load schema
        with open(self._schema_path, "r", encoding="utf-8") as f:
            schema = yaml.safe_load(f)
        self._schema_entities = [e["name"] for e in schema.get("entities", [])]

        # Load all template files
        template_files = sorted(self._templates_dir.glob("template_*.txt"))
        if not template_files:
            raise FileNotFoundError(
                f"No template files found in {self._templates_dir}"
            )
        for tf in template_files:
            with open(tf, "r", encoding="utf-8") as f:
                self._templates.append(f.read())

    @property
    def schema_entities(self) -> List[str]:
        """Return list of entity names from the schema."""
        return list(self._schema_entities)

    @property
    def template_count(self) -> int:
        """Return number of loaded templates."""
        return len(self._templates)

    def render(
        self,
        data: Dict[str, str],
        template_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Render a template with data and track entity positions.

        Args:
            data: Dictionary mapping placeholder names to generated values.
            template_index: Specific template index to use. If None, a random
                           template is selected.

        Returns:
            Dictionary with:
                - text: The rendered document text.
                - entities: List of entity dicts with label, start, end, value.
        """
        import random

        if template_index is not None:
            idx = template_index % len(self._templates)
        else:
            idx = random.randint(0, len(self._templates) - 1)

        template_text = self._templates[idx]
        renderer = _TrackingRenderer(template_text, self._schema_entities)
        return renderer.render(data)
