#!/usr/bin/env python3
"""Generate docs/www/index.md from all Markdown articles in docs/www."""

from __future__ import annotations

from pathlib import Path

DOCS_DIR = Path("docs/www")
INDEX_FILE = DOCS_DIR / "index.md"


def extract_title(md_path: Path) -> str:
    for line in md_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return md_path.stem.replace("-", " ").strip().title()


def main() -> None:
    articles = sorted(
        path
        for path in DOCS_DIR.glob("*.md")
        if path.name.lower() not in {"index.md"}
    )

    lines = [
        "# VietNerm Blog",
        "",
        "Tất cả file Markdown trong `docs/www` được tự động coi là bài viết.",
        "",
        "## Danh sách bài viết",
        "",
    ]

    if not articles:
        lines.append("_Chưa có bài viết nào._")
    else:
        for article in articles:
            title = extract_title(article)
            lines.append(f"- [{title}]({article.name})")

    INDEX_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {INDEX_FILE} with {len(articles)} articles")


if __name__ == "__main__":
    main()
