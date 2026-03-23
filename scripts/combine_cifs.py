#!/usr/bin/env python3
"""
Combine many single-structure mmCIF files (model.cif) into ONE multi-structure mmCIF
by concatenating them as separate data blocks.

This version targets:
  /mnt/gs21/scratch/garlan70/af3/outputs/OATP1B3_atorvastatin/oatp1b3_atorvastatin/**/model.cif

Run this script from:
  /mnt/gs21/scratch/garlan70/af3/scripts
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

DATA_RE = re.compile(r"^\s*data_([A-Za-z0-9_.-]*)\s*$")


def sanitize_block_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.strip("_")
    return name or "model"


def make_block_name(cif_path: Path, root: Path) -> str:
    # e.g. seed-1181819472_sample-0/model.cif -> seed-1181819472_sample-0
    rel = cif_path.relative_to(root)
    parent = rel.parent.as_posix().replace("/", "__")
    return sanitize_block_name(parent)


def rewrite_datablock(text: str, new_name: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    replaced = False

    for line in lines:
        if not replaced:
            m = DATA_RE.match(line)
            if m:
                out.append(f"data_{new_name}\n")
                replaced = True
                continue
        out.append(line)

    # If the file had no explicit data_ header, add one at the top.
    if not replaced:
        out.insert(0, f"data_{new_name}\n")

    # Ensure there's a trailing newline and a blank line between blocks.
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    if out and out[-1].strip() != "":
        out.append("\n")

    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B3_atorvastatin/oatp1b3_atorvastatin"),
        help="Directory containing seed-*/sample-* subdirs with model.cif files",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="**/model.cif",
        help="Glob pattern under --root to find CIF files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output multi-structure CIF path (default: <root>/oatp1b3_atorvastatin_all_models.cif)",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    out_path = (args.out or (root / "oatp1b3_atorvastatin_all_models.cif")).resolve()

    if not root.exists():
        raise SystemExit(f"[ERROR] Root path not found: {root}")

    cif_files = sorted(root.glob(args.pattern))
    if not cif_files:
        raise SystemExit(f"[ERROR] No CIF files found under {root} with pattern '{args.pattern}'")

    seen_names = set()
    combined_chunks = []

    for cif in cif_files:
        block_name = make_block_name(cif, root)

        # Ensure uniqueness (just in case)
        base = block_name
        i = 2
        while block_name in seen_names:
            block_name = f"{base}_{i}"
            i += 1
        seen_names.add(block_name)

        text = cif.read_text(errors="replace")
        combined_chunks.append(rewrite_datablock(text, block_name))

    out_path.write_text("".join(combined_chunks))
    print(f"[OK] Found CIFs: {len(cif_files)}")
    print(f"[OK] Wrote multi-structure CIF: {out_path}")


if __name__ == "__main__":
    main()