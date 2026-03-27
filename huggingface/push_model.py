"""
Push trained NER model to HuggingFace Hub.

Usage:
    python huggingface/push_model.py --doc cccd --repo username/phobert-cccd-ner
    python huggingface/push_model.py --doc giay_ra_vien --repo username/phobert-grv-ner
"""

import argparse
import json
import sys
from pathlib import Path
from string import Template
from typing import Optional

from huggingface_hub import HfApi, create_repo


def generate_model_card(
    doc_type: str,
    repo_id: str,
    model_dir: Path,
) -> str:
    """Generate a model card from the template.

    Args:
        doc_type: Document type identifier.
        repo_id: HuggingFace repository ID.
        model_dir: Path to the trained model directory.

    Returns:
        Rendered model card as a string.
    """
    template_path = Path(__file__).parent / "model_card.md"
    if not template_path.exists():
        return _default_model_card(doc_type, repo_id, model_dir)

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    # Load label map if available
    labels = []
    label_map_path = model_dir / "label_map.json"
    if label_map_path.exists():
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            if "label2id" in label_data:
                labels = sorted(label_data["label2id"].keys())

    config_path = model_dir / "config.json"
    base_model = "vinai/phobert-base"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            base_model = config.get("_name_or_path", base_model)

    doc_names = {
        "cccd": "Căn cước công dân (CCCD)",
        "giay_ra_vien": "Giấy ra viện",
        "vehicle_registration": "Đăng ký xe",
    }
    doc_name = doc_names.get(doc_type, doc_type)

    labels_str = "\n".join(f"- `{label}`" for label in labels if label != "O")

    template = Template(template_text)
    return template.safe_substitute(
        doc_type=doc_type,
        doc_name=doc_name,
        repo_id=repo_id,
        base_model=base_model,
        labels=labels_str,
        num_labels=len(labels),
    )


def _default_model_card(doc_type: str, repo_id: str, model_dir: Path) -> str:
    """Fallback model card when template is missing."""
    return f"""---
language: vi
tags:
  - ner
  - phobert
  - vietnamese
  - document-ai
license: mit
---

# VietNerm - {doc_type} NER Model

PhoBERT-based NER model for Vietnamese {doc_type} document entity extraction.

## Usage

```python
from vietnerm import VietNerm

ner = VietNerm(doc_type="{doc_type}", model_path="{repo_id}")
result = ner.extract("your document text here")
```
"""


def push_model(
    doc_type: str,
    repo_id: str,
    model_dir: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
) -> str:
    """Push model to HuggingFace Hub.

    Args:
        doc_type: Document type identifier.
        repo_id: HuggingFace repository ID (e.g., 'username/model-name').
        model_dir: Local model directory. Defaults to models/phobert/{doc_type}/.
        token: HuggingFace API token. Uses cached token if None.
        private: Whether to create a private repository.

    Returns:
        URL of the published model.
    """
    if model_dir is None:
        model_path = Path("models") / "phobert" / doc_type
    else:
        model_path = Path(model_dir)

    if not model_path.exists():
        print(f"[ERROR] Model directory not found: {model_path}")
        sys.exit(1)

    config_file = model_path / "config.json"
    if not config_file.exists():
        print(f"[ERROR] No config.json found in {model_path}. Is this a valid model?")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    print(f"[INFO] Creating/accessing repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private, token=token)

    # Generate and write model card
    print("[INFO] Generating model card...")
    model_card = generate_model_card(doc_type, repo_id, model_path)
    readme_path = model_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)

    # Upload all files
    print(f"[INFO] Uploading model files from {model_path}...")
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[SUCCESS] Model published to: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push VietNerm NER model to HuggingFace Hub"
    )
    parser.add_argument(
        "--doc", required=True,
        help="Document type (e.g., cccd, giay_ra_vien)",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g., username/phobert-cccd-ner)",
    )
    parser.add_argument(
        "--model-dir",
        help="Path to local model directory (default: models/phobert/{doc}/)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (default: uses cached token)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()
    push_model(
        doc_type=args.doc,
        repo_id=args.repo,
        model_dir=args.model_dir,
        token=args.token,
        private=args.private,
    )


if __name__ == "__main__":
    main()
