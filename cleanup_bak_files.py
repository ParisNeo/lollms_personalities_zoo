#!/usr/bin/env python3
"""
Removes all .bak files recursively from the repository.
"""
from pathlib import Path

root = Path(__file__).parent.resolve()
count = 0

for bak in root.rglob("*.bak"):
    print(f"Removing: {bak.relative_to(root)}")
    bak.unlink()
    count += 1

print(f"\nDone. Removed {count} .bak file(s).")
