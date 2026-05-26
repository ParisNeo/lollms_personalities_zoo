#!/usr/bin/env python3
"""
SOUL.md Migration Script

Scans the lollms_personalities_zoo repository and rewrites every SOUL.md
from the legacy multi-section Markdown format into a HuggingFace-style
YAML-frontmatter format.

Usage:
    python migrate_soul_format.py              # Execute migration
    python migrate_soul_format.py --dry-run    # Preview changes
"""
import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML is required. Install it with: pip install pyyaml")
    sys.exit(1)


def extract_section(content: str, header: str) -> str:
    """
    Extract text under a `## Header` until the next `## ` header or end of file.
    The match is case-insensitive.
    """
    pattern = rf"## {re.escape(header)}\s*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def extract_metadata(content: str) -> dict:
    """
    Extract the YAML block inside the `## Metadata` section.
    """
    section = extract_section(content, "Metadata")
    if not section:
        return {}

    # Remove optional markdown fences (```yaml ... ```)
    cleaned = re.sub(r"^```(?:yaml)?\s*\n", "", section)
    cleaned = re.sub(r"\n```\s*$", "", cleaned)

    try:
        return yaml.safe_load(cleaned) or {}
    except yaml.YAMLError:
        return {}


def build_frontmatter(meta: dict, description: str, persona_path: Path) -> str:
    """
    Construct the YAML frontmatter string with keys in the required order.
    """
    # Fallbacks derived from directory layout: category/persona/SOUL.md
    category = meta.get("category") or persona_path.parent.parent.name
    name = meta.get("name") or persona_path.parent.name
    version = str(meta.get("version") or "1.0.0")
    author = str(meta.get("author") or "ParisNeo")

    # Ensure temperature is a float
    raw_temp = meta.get("model_parameters", {}).get("temperature", 0.7)
    try:
        temperature = float(raw_temp)
    except (ValueError, TypeError):
        temperature = 0.7

    # Fallback description from metadata if section was empty
    if not description:
        description = str(meta.get("description") or "")

    data = {
        "name": name,
        "author": author,
        "version": version,
        "category": category,
        "temperature": temperature,
        "description": description,
    }

    dumped = yaml.safe_dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=float("inf"),
    ).strip()

    return f"---\n{dumped}\n---"


def migrate_soul(path: Path, dry_run: bool = False) -> bool:
    content = path.read_text(encoding="utf-8")

    # Skip if already migrated (starts with YAML frontmatter)
    if content.strip().startswith("---"):
        print(f"[SKIP] Already migrated: {path}")
        return False

    description = extract_section(content, "Description")
    conditioning = extract_section(content, "Conditioning")
    welcome = extract_section(content, "Welcome Message")
    meta = extract_metadata(content)

    frontmatter = build_frontmatter(meta, description, path)

    body_parts = []
    if conditioning:
        body_parts.append(conditioning)
    if welcome:
        body_parts.append(welcome)

    body = "\n\n".join(body_parts)
    new_content = f"{frontmatter}\n\n{body}\n"

    if dry_run:
        print(f"[DRY-RUN] {path}")
        print("-" * 40)
        print(new_content[:800])
        print("-" * 40)
        return True

    # Backup original
    backup_path = path.with_suffix(".md.bak")
    backup_path.write_text(content, encoding="utf-8")

    path.write_text(new_content, encoding="utf-8")
    print(f"[MIGRATED] {path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SOUL.md files to YAML-frontmatter format."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print previews without writing changes.",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    souls = sorted(p for p in root.rglob("SOUL.md") if not p.name.endswith(".bak"))

    migrated = 0
    skipped = 0
    for soul in souls:
        if migrate_soul(soul, dry_run=args.dry_run):
            migrated += 1
        else:
            skipped += 1

    print(f"\nDone. Migrated: {migrated}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
