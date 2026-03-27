#!/usr/bin/env python3
"""Discover document types from registry/documents.yaml.

Reads the registry and outputs a JSON array of doc type keys.
Used by GitHub Actions matrix strategy to dynamically build jobs
for each document type.

Usage:
    python .github/workflows/scripts/discover_docs.py
    # Output: ["cccd", "giay_ra_vien", "vehicle_registration"]

    python .github/workflows/scripts/discover_docs.py --filter cccd
    # Output: ["cccd"]
"""

import argparse
import json
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Discover doc types from registry")
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Return only this doc type (if it exists in registry)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to registry YAML (default: auto-detect)",
    )
    args = parser.parse_args()

    # Find registry file — walk up from script location to repo root
    if args.registry:
        registry_path = Path(args.registry)
    else:
        # Script is at .github/workflows/scripts/ — repo root is 3 levels up
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent.parent
        registry_path = repo_root / "registry" / "documents.yaml"

    if not registry_path.exists():
        print(f"Error: Registry not found at {registry_path}", file=sys.stderr)
        sys.exit(1)

    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    doc_types = list(registry.get("documents", {}).keys())

    if not doc_types:
        print("Error: No document types found in registry", file=sys.stderr)
        sys.exit(1)

    # Optionally filter to a single doc type
    if args.filter:
        if args.filter not in doc_types:
            print(
                f"Error: '{args.filter}' not in registry. Available: {doc_types}",
                file=sys.stderr,
            )
            sys.exit(1)
        doc_types = [args.filter]

    # Output as JSON array (consumed by GitHub Actions matrix)
    print(json.dumps(doc_types))


if __name__ == "__main__":
    main()
