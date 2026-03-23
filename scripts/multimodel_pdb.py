#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import sys

SEED_SAMPLE_RE = re.compile(r"seed-(\d+)_sample-(\d+)")


def extract_seed_sample(path: Path):
    """
    Return (seed, sample) parsed from .../seed-<seed>_sample-<sample>/model.cif
    Used for stable sorting.
    """
    m = SEED_SAMPLE_RE.search(str(path))
    if not m:
        return (10**30, 10**30)
    return (int(m.group(1)), int(m.group(2)))


def cif_to_pdb_string(cif_path: Path):
    try:
        import gemmi  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: gemmi.\n"
            "Install with:\n"
            "  conda install -c conda-forge gemmi\n"
            "or\n"
            "  python -m pip install gemmi"
        ) from e

    st = gemmi.read_structure(str(cif_path))
    pdb = st.make_pdb_string()

    # Remove trailing END so we can wrap with MODEL/ENDMDL cleanly
    lines = pdb.splitlines(True)  # keepends
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].startswith("END"):
        lines.pop()

    return "".join(lines).rstrip("\n") + "\n"


def write_multimodel_pdb(cif_files, out_pdb: Path):
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w") as fh:
        for i, cif_path in enumerate(cif_files, start=1):
            seed, sample = extract_seed_sample(cif_path)
            fh.write(f"MODEL     {i:>4d}\n")
            fh.write(f"REMARK   1 SOURCE {cif_path}\n")
            fh.write(f"REMARK   1 SEED {seed} SAMPLE {sample}\n")
            fh.write(cif_to_pdb_string(cif_path))
            fh.write("ENDMDL\n")
        fh.write("END\n")


def main():
    ap = argparse.ArgumentParser(
        description="Create one multi-model PDB per OATP from AF3 output CIFs (all samples)."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/gs21/scratch/garlan70/af3"),
        help="AF3 root dir (default: /mnt/gs21/scratch/garlan70/af3)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output PDBs if they already exist",
    )
    ap.add_argument(
        "--first-subdir-only",
        action="store_true",
        help="Only search the first directory under outputs/OATPxxxx/ (matches your current layout).",
    )
    args = ap.parse_args()

    root = args.root.resolve()
    outputs = (root / "outputs").resolve()
    if not outputs.exists():
        print(f"ERROR: outputs dir not found: {outputs}", file=sys.stderr)
        sys.exit(1)

    oatp_dirs = [p for p in outputs.iterdir() if p.is_dir()]
    if not oatp_dirs:
        print(f"ERROR: no directories found under: {outputs}", file=sys.stderr)
        sys.exit(1)

    made_any = False

    for oatp_dir in sorted(oatp_dirs):
        oatp_name = oatp_dir.name

        search_roots = []
        if args.first_subdir_only:
            subs = [p for p in oatp_dir.iterdir() if p.is_dir()]
            if not subs:
                print(f"[skip] {oatp_name}: no subdirectory under {oatp_dir}")
                continue
            search_roots = [sorted(subs)[0]]
        else:
            # Search anywhere under outputs/OATPxxxx/
            search_roots = [oatp_dir]

        cif_files = []
        for sr in search_roots:
            cif_files.extend(sr.rglob("seed-*_sample-*/model.cif"))

        if not cif_files:
            print(f"[skip] {oatp_name}: no CIFs found (seed-*_sample-*/model.cif)")
            continue

        # Deduplicate + sort
        cif_files = sorted(set(cif_files), key=extract_seed_sample)

        out_pdb = oatp_dir / f"{oatp_name}_multimodel.pdb"
        if out_pdb.exists() and not args.overwrite:
            print(f"[skip] {oatp_name}: output exists (use --overwrite): {out_pdb}")
            continue

        print(f"[write] {oatp_name}: {len(cif_files)} models -> {out_pdb}")
        write_multimodel_pdb(cif_files, out_pdb)
        made_any = True

    if not made_any:
        print("No multi-model PDBs were written (nothing matched).", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
