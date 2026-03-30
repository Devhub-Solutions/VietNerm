"""OCR noise simulation engine for Vietnamese text.

Simulates common OCR errors including character substitution, deletion,
insertion, Vietnamese diacritics errors, and random spacing issues.
"""

import random
from typing import Any, Dict, List, Tuple

# Common OCR noise tokens (from original codebase)
OCR_NOISE_TOKENS: List[str] = [
    "Star", "It", "AND", "OF", "THE", "MS:",
    "REFITTED", "ZORX", "riston", "Therming",
    "windswy", "Sentifier", "Cesm", "Emok",
    "Contrước", "Trong", "Thuật", "CỔ PHẦN",
    "j5072810750", "Swoodbản",
    "1996", "1970", "1971", "1992", "1988", "2001",
    "0", "triển", "THỊ", "X", "6", "19", "TẾ", "C", "Số I No.",
]

# Vietnamese diacritics confusion pairs (common OCR errors)
DIACRITICS_PAIRS: List[Tuple[str, str]] = [
    ("ă", "a"), ("â", "a"), ("ê", "e"), ("ô", "o"), ("ơ", "o"),
    ("ư", "u"), ("đ", "d"),
    ("Ă", "A"), ("Â", "A"), ("Ê", "E"), ("Ô", "O"), ("Ơ", "O"),
    ("Ư", "U"), ("Đ", "D"),
    ("á", "a"), ("à", "a"), ("ả", "a"), ("ã", "a"), ("ạ", "a"),
    ("ắ", "ă"), ("ằ", "ă"), ("ẳ", "ă"), ("ẵ", "ă"), ("ặ", "ă"),
    ("ấ", "â"), ("ầ", "â"), ("ẩ", "â"), ("ẫ", "â"), ("ậ", "â"),
    ("é", "e"), ("è", "e"), ("ẻ", "e"), ("ẽ", "e"), ("ẹ", "e"),
    ("ế", "ê"), ("ề", "ê"), ("ể", "ê"), ("ễ", "ê"), ("ệ", "ê"),
    ("í", "i"), ("ì", "i"), ("ỉ", "i"), ("ĩ", "i"), ("ị", "i"),
    ("ó", "o"), ("ò", "o"), ("ỏ", "o"), ("õ", "o"), ("ọ", "o"),
    ("ố", "ô"), ("ồ", "ô"), ("ổ", "ô"), ("ỗ", "ô"), ("ộ", "ô"),
    ("ớ", "ơ"), ("ờ", "ơ"), ("ở", "ơ"), ("ỡ", "ơ"), ("ợ", "ơ"),
    ("ú", "u"), ("ù", "u"), ("ủ", "u"), ("ũ", "u"), ("ụ", "u"),
    ("ứ", "ư"), ("ừ", "ư"), ("ử", "ư"), ("ữ", "ư"), ("ự", "ư"),
    ("ý", "y"), ("ỳ", "y"), ("ỷ", "y"), ("ỹ", "y"), ("ỵ", "y"),
]

# Character-level substitution map for OCR errors
CHAR_SUBSTITUTIONS: Dict[str, List[str]] = {
    "0": ["O", "o", "Q"],
    "O": ["0", "Q"],
    "o": ["0", "O"],
    "1": ["l", "I", "i", "|"],
    "l": ["1", "I", "|"],
    "I": ["1", "l", "|"],
    "5": ["S", "s"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "6": ["G", "b"],
    "9": ["q"],
    "2": ["Z", "z"],
    "Z": ["2"],
}


class NoiseEngine:
    """Applies OCR-like noise to text while tracking entity position shifts.

    Supports configurable noise level from 0.0 (no noise) to 1.0 (heavy noise).
    """

    def __init__(self, noise_level: float = 0.1) -> None:
        """Initialize the noise engine.

        Args:
            noise_level: Float from 0.0 to 1.0 controlling noise intensity.
        """
        self._noise_level = max(0.0, min(1.0, noise_level))

    @property
    def noise_level(self) -> float:
        return self._noise_level

    @noise_level.setter
    def noise_level(self, value: float) -> None:
        self._noise_level = max(0.0, min(1.0, value))

    def apply(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Apply noise to a rendered record, updating entity positions.

        Args:
            record: Dictionary with 'text' and 'entities' from TemplateEngine.

        Returns:
            New dictionary with noised text and updated entity positions.
        """
        if self._noise_level <= 0.0:
            return record

        text = record["text"]
        entities = [dict(e) for e in record["entities"]]  # deep copy

        # Sort entities by start position
        entities.sort(key=lambda e: e["start"])

        # Build a map of protected ranges (entity values)
        protected_ranges = [(e["start"], e["end"]) for e in entities]

        # Apply noise to non-entity regions
        noised_text, entity_offsets = self._apply_char_noise(
            text, protected_ranges
        )

        # Update entity positions based on offsets
        updated_entities = []
        for entity in entities:
            offset = entity_offsets.get(entity["start"], 0)
            new_start = entity["start"] + offset
            new_end = new_start + len(entity["value"])
            updated_entities.append({
                "label": entity["label"],
                "start": new_start,
                "end": new_end,
                "value": entity["value"],
            })

        # Optionally add OCR noise tokens at line boundaries
        if random.random() < self._noise_level * 0.5:
            noised_text, updated_entities = self._add_noise_tokens(
                noised_text, updated_entities
            )

        return {
            "text": noised_text,
            "entities": updated_entities,
        }

    def _apply_char_noise(
        self,
        text: str,
        protected_ranges: List[Tuple[int, int]],
    ) -> Tuple[str, Dict[int, int]]:
        """Apply character-level noise outside protected entity ranges.

        Returns:
            Tuple of (noised text, mapping from original entity starts to offset).
        """
        result_chars: List[str] = []
        cumulative_offset = 0
        # Track offsets at each protected range start
        entity_offsets: Dict[int, int] = {}

        # Mark the start of each protected range
        range_starts = {r[0] for r in protected_ranges}

        def _is_protected(pos: int) -> bool:
            for start, end in protected_ranges:
                if start <= pos < end:
                    return True
            return False

        i = 0
        while i < len(text):
            # Record offset at entity boundaries
            if i in range_starts:
                entity_offsets[i] = cumulative_offset

            if _is_protected(i):
                result_chars.append(text[i])
                i += 1
                continue

            char = text[i]

            # Newline corruption (OCR often merges/splits lines unexpectedly)
            if char == "\n" and random.random() < self._noise_level * 0.12:
                if random.random() < 0.5:
                    # Merge two lines directly
                    cumulative_offset -= 1
                    i += 1
                    continue
                # Replace newline with a single space
                result_chars.append(" ")
                i += 1
                continue

            # Character substitution
            if (random.random() < self._noise_level * 0.15
                    and char in CHAR_SUBSTITUTIONS):
                replacement = random.choice(CHAR_SUBSTITUTIONS[char])
                result_chars.append(replacement)
                i += 1
                continue

            # Diacritics error
            if random.random() < self._noise_level * 0.1:
                for original, replacement in DIACRITICS_PAIRS:
                    if char == original:
                        result_chars.append(replacement)
                        i += 1
                        break
                else:
                    # No diacritics match, proceed normally
                    pass

                if i > 0 and result_chars and result_chars[-1] != text[i - 1]:
                    # Already handled
                    continue

            # Character deletion (including occasional whitespace deletion)
            if random.random() < self._noise_level * 0.05 and char != "\n":
                cumulative_offset -= 1
                i += 1
                continue

            # Character insertion
            if random.random() < self._noise_level * 0.03 and char not in "\n":
                insert_char = random.choice("abcdefghijklmnopqrstuvwxyz0123456789")
                result_chars.append(insert_char)
                cumulative_offset += 1

            # Spacing issues
            if random.random() < self._noise_level * 0.05:
                if char == " ":
                    if random.random() < 0.6:
                        # Double space
                        result_chars.append("  ")
                        cumulative_offset += 1
                    else:
                        # Remove separator space (e.g., "GIẤY RA VIỆN" -> "GIẤY RAVIỆN")
                        cumulative_offset -= 1
                    i += 1
                    continue

            result_chars.append(char)
            i += 1

        # Fill in offsets for any entity starts not yet recorded
        for start, _ in protected_ranges:
            if start not in entity_offsets:
                entity_offsets[start] = cumulative_offset

        return "".join(result_chars), entity_offsets

    def _add_noise_tokens(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Insert random OCR noise tokens at line boundaries."""
        lines = text.split("\n")
        new_lines: List[str] = []
        inserted_before: Dict[int, int] = {}  # line_index -> chars inserted

        for i, line in enumerate(lines):
            if random.random() < self._noise_level * 0.25 and line.strip():
                noise_token = random.choice(OCR_NOISE_TOKENS)
                if random.random() < 0.5:
                    new_line = noise_token + " " + line
                    inserted_before[i] = len(noise_token) + 1
                else:
                    new_line = line + " " + noise_token
                new_lines.append(new_line)
            else:
                new_lines.append(line)


        new_text = "\n".join(new_lines)

        # Recalculate entity positions by finding values in new text
        # Use starting positions to avoid duplicate finding issues (e.g. 'Nam' in different fields)
        updated_entities = []
        search_pos = 0
        for entity in entities:
            value = entity["value"]
            # Search for the value starting from the last known end point
            idx = new_text.find(value, search_pos)
            if idx != -1:
                updated_entities.append({
                    "label": entity["label"],
                    "start": idx,
                    "end": idx + len(value),
                    "value": value,
                })
                search_pos = idx + len(value)
            else:
                # Value not found (fallback to original)
                updated_entities.append(entity)

        return new_text, updated_entities
