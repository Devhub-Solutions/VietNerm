"""CLI for generating synthetic NER datasets from document templates.

Usage:
    python synthetic/generate_dataset.py --doc cccd --size 50000 --noise_level 0.1
    python synthetic/generate_dataset.py --doc giay_ra_vien --size 2000
    python synthetic/generate_dataset.py --doc vehicle_registration --size 1000
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Project root — also add to sys.path so imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_registry() -> Dict[str, Any]:
    """Load the document registry."""
    registry_path = PROJECT_ROOT / "registry" / "documents.yaml"
    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_generator(generator_name: str):
    """Get a generator instance by name."""
    from synthetic.generators import GENERATOR_MAP

    if generator_name not in GENERATOR_MAP:
        raise ValueError(
            f"Unknown generator: {generator_name}. "
            f"Available: {list(GENERATOR_MAP.keys())}"
        )
    return GENERATOR_MAP[generator_name]()


# ═══════════════════════════════════════════
# BIO CONVERSION (migrated from dataset/builder.py and raw_code)
# ═══════════════════════════════════════════

def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text by whitespace and track character offsets.

    Replaces newlines with spaces, then splits on whitespace.
    Returns list of (token, start, end) tuples.
    """
    flat = text.replace("\n", " ")
    result: List[Tuple[str, int, int]] = []
    pos = 0
    for tok in flat.split(" "):
        if not tok:
            pos += 1
            continue
        start = flat.find(tok, pos)
        if start == -1:
            pos += len(tok) + 1
            continue
        end = start + len(tok)
        result.append((tok, start, end))
        pos = end + 1
    return result


def assign_bio_labels(
    tokens_with_pos: List[Tuple[str, int, int]],
    entities: List[Dict[str, Any]],
) -> List[str]:
    """Assign BIO labels to tokens based on entity character spans.

    Each entity dict must have: label, start, end.
    Produces B-{label}/I-{label} tags.
    """
    labels = ["O"] * len(tokens_with_pos)

    # Sort entities by start position
    sorted_entities = sorted(entities, key=lambda e: e["start"])

    for entity in sorted_entities:
        span_start = entity["start"]
        span_end = entity["end"]
        tag_name = entity["label"]
        first_in_span = True

        for i, (_, tok_start, tok_end) in enumerate(tokens_with_pos):
            if tok_end <= span_start or tok_start >= span_end:
                continue
            if labels[i] == "O":
                labels[i] = f"B-{tag_name}" if first_in_span else f"I-{tag_name}"
                first_in_span = False

    return labels


def record_to_bio_sample(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a rendered record (with text and entities) to BIO format.

    Args:
        record: Dict with 'text' and 'entities' from template engine.

    Returns:
        Dict with 'tokens' (list of str) and 'ner_tags' (list of BIO labels).
    """
    text = record["text"]
    entities = record["entities"]

    tokens_with_pos = tokenize_with_offsets(text)
    if not tokens_with_pos:
        return {"tokens": [], "ner_tags": []}

    tokens = [t for t, _, _ in tokens_with_pos]
    tags = assign_bio_labels(tokens_with_pos, entities)

    return {"tokens": tokens, "ner_tags": tags}


def get_label_list(samples: List[Dict[str, Any]]) -> List[str]:
    """Extract sorted unique label list from samples, with O first."""
    tag_set = {"O"}
    for s in samples:
        for t in s["ner_tags"]:
            tag_set.add(t)
    return ["O"] + sorted(tag_set - {"O"})


# ═══════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════

def generate_dataset(
    doc_type: str,
    size: int = 50000,
    noise_level: float = 0.1,
    test_split: float = 0.1,
) -> None:
    """Generate a complete NER dataset for a document type.

    Steps:
        1. Load registry to find templates, schema, generator
        2. Generate raw records
        3. Convert to NER BIO format
        4. Split train/test
        5. Save to datasets/

    Args:
        doc_type: Document type key from registry (e.g., 'cccd').
        size: Number of samples to generate.
        noise_level: OCR noise level (0.0 to 1.0).
        test_split: Fraction of data for test set.
    """
    registry = load_registry()
    if doc_type not in registry["documents"]:
        available = list(registry["documents"].keys())
        print(f"Error: Unknown document type '{doc_type}'. Available: {available}")
        sys.exit(1)

    doc_config = registry["documents"][doc_type]
    print(f"==> Generating dataset for: {doc_config['name']} ({doc_type})")
    print(f"    Size: {size}, Noise level: {noise_level}")

    # Initialize generator
    generator = get_generator(doc_config["generator"])
    print(f"    Generator: {doc_config['generator']}")

    # Initialize template engine
    from synthetic.template_engine import TemplateEngine

    templates_dir = PROJECT_ROOT / doc_config["templates"]
    schema_path = PROJECT_ROOT / doc_config["schema"]
    template_engine = TemplateEngine(templates_dir, schema_path)
    print(f"    Templates: {template_engine.template_count} loaded")
    print(f"    Schema entities: {template_engine.schema_entities}")

    # Initialize noise engine
    from synthetic.noise_engine import NoiseEngine

    noise_engine = NoiseEngine(noise_level=noise_level)

    # Generate raw records
    print(f"\n==> Generating {size} raw records...")
    raw_records: List[Dict[str, Any]] = []
    rendered_records: List[Dict[str, Any]] = []

    for i in range(size):
        data = generator.generate()
        raw_records.append(data)

        rendered = template_engine.render(data)
        if noise_level > 0:
            rendered = noise_engine.apply(rendered)
        rendered_records.append(rendered)

        if (i + 1) % 10000 == 0:
            print(f"    Generated {i + 1}/{size}")

    # Save raw records
    raw_dir = PROJECT_ROOT / "datasets" / "raw" / doc_type
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "raw_records.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_records, f, ensure_ascii=False, indent=2)
    print(f"    Saved raw records to {raw_path}")

    # Convert to BIO format
    print(f"\n==> Converting to NER BIO format...")
    bio_samples: List[Dict[str, Any]] = []
    for record in rendered_records:
        sample = record_to_bio_sample(record)
        if sample["tokens"]:
            bio_samples.append(sample)

    print(f"    Valid BIO samples: {len(bio_samples)}/{size}")

    # Shuffle and split
    random.shuffle(bio_samples)
    split_idx = int(len(bio_samples) * (1 - test_split))
    train_samples = bio_samples[:split_idx]
    test_samples = bio_samples[split_idx:]

    # Get label list
    labels = get_label_list(bio_samples)
    label2id = {label: idx for idx, label in enumerate(labels)}

    print(f"    Train: {len(train_samples)}, Test: {len(test_samples)}")
    print(f"    Labels ({len(labels)}): {labels}")

    # Save NER dataset
    ner_dir = PROJECT_ROOT / "datasets" / "ner" / doc_type
    ner_dir.mkdir(parents=True, exist_ok=True)

    train_path = ner_dir / "train.json"
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)

    test_path = ner_dir / "test.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)

    labels_path = ner_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "label2id": label2id,
            "id2label": {v: k for k, v in label2id.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n==> Dataset saved to {ner_dir}/")
    print(f"    train.json: {len(train_samples)} samples")
    print(f"    test.json: {len(test_samples)} samples")
    print(f"    labels.json: {len(labels)} labels")
    print("==> Done!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic NER dataset for Vietnamese documents"
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Document type from registry (e.g., cccd, giay_ra_vien, vehicle_registration)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50000,
        help="Number of samples to generate (default: 50000)",
    )
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="OCR noise level 0.0-1.0 (default: 0.1)",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Fraction for test set (default: 0.1)",
    )
    args = parser.parse_args()

    generate_dataset(
        doc_type=args.doc,
        size=args.size,
        noise_level=args.noise_level,
        test_split=args.test_split,
    )


if __name__ == "__main__":
    main()
