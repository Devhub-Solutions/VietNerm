"""
Push NER dataset to HuggingFace Hub.

Usage:
    python huggingface/push_dataset.py --doc cccd --repo username/vietnerm-cccd-dataset
    python huggingface/push_dataset.py --doc giay_ra_vien --repo username/vietnerm-grv-dataset
"""

import argparse
import json
import sys
from pathlib import Path
from string import Template
from typing import Optional


def _build_labels_table(labels: list[str]) -> str:
    """Build a Markdown table of BIO labels."""
    if not labels:
        return "_No labels file found._"
    rows = ["| Label | Type |", "|-------|------|"]
    for label in sorted(labels):
        if label == "O":
            rows.append(f"| `{label}` | Outside |")
        elif label.startswith("B-"):
            rows.append(f"| `{label}` | Begin — {label[2:]} |")
        elif label.startswith("I-"):
            rows.append(f"| `{label}` | Inside — {label[2:]} |")
        else:
            rows.append(f"| `{label}` | — |")
    return "\n".join(rows)


def _build_mockup_example(dataset_dir: Path) -> str:
    """Extract a representative sample from train.json for the README."""
    train_path = dataset_dir / "train.json"
    if not train_path.exists():
        return '{"tokens": ["..."], "ner_tags": ["..."]}'
    with open(train_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    # Pick a sample that has at least one non-O tag
    chosen = None
    for s in samples:
        tags = s.get("ner_tags", s.get("labels", []))
        if any(t != "O" for t in tags):
            chosen = s
            break
    if chosen is None and samples:
        chosen = samples[0]
    if chosen is None:
        return '{"tokens": ["..."], "ner_tags": ["..."]}'

    tokens = chosen.get("tokens", [])
    tags = chosen.get("ner_tags", chosen.get("labels", []))
    # Truncate to first 20 tokens for readability
    tokens = tokens[:20]
    tags = tags[:20]
    example = {"tokens": tokens, "ner_tags": tags}
    return json.dumps(example, ensure_ascii=False, indent=2)


def generate_dataset_card(
    doc_type: str,
    repo_id: str,
    dataset_dir: Path,
    hf_username: str = "ngocthanhdoan",
) -> str:
    """Generate a dataset card from template.

    Args:
        doc_type: Document type identifier.
        repo_id: HuggingFace repository ID.
        dataset_dir: Path to the dataset directory.
        hf_username: HuggingFace username for companion model link.

    Returns:
        Dataset card as a string.
    """
    doc_names = {
        "cccd": "Căn cước công dân (CCCD)",
        "giay_ra_vien": "Giấy ra viện",
        "vehicle_registration": "Đăng ký xe",
    }
    doc_name = doc_names.get(doc_type, doc_type)

    # Count samples
    train_count = 0
    test_count = 0
    train_path = dataset_dir / "train.json"
    test_path = dataset_dir / "test.json"
    if train_path.exists():
        with open(train_path, "r", encoding="utf-8") as f:
            train_count = len(json.load(f))
    if test_path.exists():
        with open(test_path, "r", encoding="utf-8") as f:
            test_count = len(json.load(f))

    # Load labels
    labels = []
    labels_path = dataset_dir / "labels.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    labels_table = _build_labels_table(labels)
    mockup_example = _build_mockup_example(dataset_dir)

    # Load template
    template_path = Path(__file__).parent / "dataset_card.md"
    if not template_path.exists():
        # Fallback minimal card
        return f"# VietNerm — {doc_name} NER Dataset\n\nSynthetic NER dataset.\n"

    with open(template_path, "r", encoding="utf-8") as f:
        template_text = f.read()

    tmpl = Template(template_text)
    return tmpl.safe_substitute(
        doc_type=doc_type,
        doc_name=doc_name,
        repo_id=repo_id,
        hf_username=hf_username,
        train_count=train_count,
        test_count=test_count,
        labels_table=labels_table,
        mockup_example=mockup_example,
    )


def _squash_repo_history(
    repo_id: str,
    repo_type: str = "dataset",
    token: Optional[str] = None,
) -> None:
    """Squash all commits into one, keeping only the latest version."""
    from huggingface_hub import HfApi  # noqa: PLC0415
    api = HfApi(token=token)
    try:
        commits_before = api.list_repo_commits(repo_id, repo_type=repo_type, token=token)
        n_before = len(commits_before)
        if n_before <= 1:
            print(f"[INFO] History already clean ({n_before} commit) — skip squash")
            return
        print(f"[INFO] Squashing {n_before} commits into 1...")
        api.super_squash_history(
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="chore: squash history — keep latest version only",
            token=token,
        )
        print(f"[INFO] History squashed: {n_before} commits → 1")
    except Exception as e:
        print(f"[WARNING] Could not squash history for {repo_id}: {e}")


def push_dataset(
    doc_type: str,
    repo_id: str,
    dataset_dir: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
    hf_username: str = "ngocthanhdoan",
) -> str:
    """Push dataset to HuggingFace Hub.

    Args:
        doc_type: Document type identifier.
        repo_id: HuggingFace repository ID.
        dataset_dir: Local dataset directory. Defaults to datasets/ner/{doc_type}/.
        token: HuggingFace API token.
        private: Whether to create a private repository.
        hf_username: HuggingFace username for companion model link.

    Returns:
        URL of the published dataset.
    """
    from huggingface_hub import HfApi, create_repo  # noqa: PLC0415

    if dataset_dir is None:
        ds_path = Path("datasets") / "ner" / doc_type
    else:
        ds_path = Path(dataset_dir)

    if not ds_path.exists():
        print(f"[ERROR] Dataset directory not found: {ds_path}")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo
    print(f"[INFO] Creating/accessing dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private, token=token)

    # Generate dataset card from template
    print("[INFO] Generating dataset card from template...")
    dataset_card = generate_dataset_card(doc_type, repo_id, ds_path, hf_username=hf_username)
    readme_path = ds_path / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(dataset_card)
    print(f"[INFO] README.md written ({len(dataset_card)} chars)")

    # Upload
    print(f"[INFO] Uploading dataset from {ds_path}...")
    api.upload_folder(
        folder_path=str(ds_path),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    # Squash history: giữ chỉ version mới nhất, xóa tất cả commit cũ
    print("[INFO] Squashing dataset repo history...")
    _squash_repo_history(repo_id, repo_type="dataset", token=token)

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[SUCCESS] Dataset published to: {url}")
    return url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push VietNerm NER dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--doc", required=True,
        help="Document type (e.g., cccd, giay_ra_vien)",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g., username/vietnerm-cccd-dataset)",
    )
    parser.add_argument(
        "--dataset-dir",
        help="Path to dataset directory (default: datasets/ner/{doc}/)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (default: uses cached token)",
    )
    parser.add_argument(
        "--hf-username", default="ngocthanhdoan",
        help="HuggingFace username for companion model link",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()
    push_dataset(
        doc_type=args.doc,
        repo_id=args.repo,
        dataset_dir=args.dataset_dir,
        token=args.token,
        private=args.private,
        hf_username=args.hf_username,
    )


if __name__ == "__main__":
    main()
