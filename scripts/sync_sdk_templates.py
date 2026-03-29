"""Sync root templates/ into sdk/vietnerm/templates/ before SDK build.

Run this script from repository root:

    python scripts/sync_sdk_templates.py
"""

from pathlib import Path
import shutil


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "templates"
    dst_root = repo_root / "sdk" / "vietnerm" / "templates"

    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing source templates dir: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    src_doc_types = {
        p.name for p in src_root.iterdir() if p.is_dir()
    }

    # Remove stale doc types in sdk templates
    for existing in dst_root.iterdir():
        if existing.is_dir() and existing.name not in src_doc_types:
            shutil.rmtree(existing)

    # Copy each doc type folder
    for doc_type in sorted(src_doc_types):
        src_dir = src_root / doc_type
        dst_dir = dst_root / doc_type
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)

    print(
        f"Synced {len(src_doc_types)} template folders "
        f"from {src_root} -> {dst_root}"
    )


if __name__ == "__main__":
    main()

