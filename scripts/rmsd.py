#!/usr/bin/env python3
"""
RMSD Alignment Script — BioPython implementation
Parses mmCIF files properly, aligns on CA atoms shared between reference
and model (by sequence position), and computes RMSD via SVD superposition.

Dependencies:
    pip install biopython

Usage:
    python /mnt/gs21/scratch/garlan70/af3/scripts/rmsd.py
"""

import os
import glob
import csv
import sys
import numpy as np

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Superimposer import Superimposer
    from Bio import pairwise2
except ImportError:
    sys.exit(
        "ERROR: BioPython not found.\n"
        "Install with:  pip install biopython"
    )

# ── Configuration ─────────────────────────────────────────────────────────────

PAIRS = [
    {
        "parent_dir": "/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B1_inward/oatp1b1_inward",
        "reference":  "/mnt/gs21/scratch/garlan70/af3/inputs/structures/OATP1B1_inward.cif",
        "label":      "OATP1B1_inward",
    },
    {
        "parent_dir": "/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B1_outward/oatp1b1_outward",
        "reference":  "/mnt/gs21/scratch/garlan70/af3/inputs/structures/OATP1B1_outward.cif",
        "label":      "OATP1B1_outward",
    },
    {
        "parent_dir": "/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B3_inward/oatp1b3_inward",
        "reference":  "/mnt/gs21/scratch/garlan70/af3/inputs/structures/OATP1B3_inward.cif",
        "label":      "OATP1B3_inward",
    },
    {
        "parent_dir": "/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B3_outward/oatp1b3_outward",
        "reference":  "/mnt/gs21/scratch/garlan70/af3/inputs/structures/OATP1B3_outward.cif",
        "label":      "OATP1B3_outward",
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

PARSER = MMCIFParser(QUIET=True)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def get_ca_atoms(structure):
    """
    Return an ordered list of (resseq, resname, CA_atom) for all standard
    residues with a CA atom, from the first model/chain.
    Handles structures with a single chain A or picks the longest chain.
    """
    model  = structure[0]
    chains = list(model.get_chains())

    # Pick chain A if present, otherwise the longest chain
    chain = None
    for ch in chains:
        if ch.id == "A":
            chain = ch
            break
    if chain is None:
        chain = max(chains, key=lambda c: sum(1 for r in c if "CA" in r))

    ca_list = []
    for residue in chain:
        # Skip HETATM records (water, ligands)
        if residue.id[0] != " ":
            continue
        if "CA" not in residue:
            continue
        resseq  = residue.id[1]
        resname = residue.resname.strip()
        ca_list.append((resseq, resname, residue["CA"]))

    return ca_list


def sequence_from_ca_list(ca_list):
    """Return one-letter sequence string from CA atom list."""
    return "".join(THREE_TO_ONE.get(r[1], "X") for r in ca_list)


def align_sequences_and_get_paired_atoms(ref_ca, mob_ca):
    """
    Sequence-align the two CA lists and return paired (ref_atoms, mob_atoms)
    lists containing only matched, gap-free positions.
    Uses BioPython pairwise2 global alignment.
    """
    ref_seq = sequence_from_ca_list(ref_ca)
    mob_seq = sequence_from_ca_list(mob_ca)

    # Global alignment: match=2, mismatch=-1, gap_open=-5, gap_extend=-0.5
    alignments = pairwise2.align.globalms(
        ref_seq, mob_seq,
        2, -1, -5, -0.5,
        one_alignment_only=True,
    )

    if not alignments:
        raise ValueError("Sequence alignment failed — no alignment found.")

    aln_ref, aln_mob = alignments[0].seqA, alignments[0].seqB

    ref_atoms = []
    mob_atoms = []
    ref_idx = 0
    mob_idx = 0

    for r, m in zip(aln_ref, aln_mob):
        if r != "-" and m != "-":
            ref_atoms.append(ref_ca[ref_idx][2])
            mob_atoms.append(mob_ca[mob_idx][2])
        if r != "-":
            ref_idx += 1
        if m != "-":
            mob_idx += 1

    return ref_atoms, mob_atoms


def compute_rmsd(ref_atoms, mob_atoms):
    """
    Superpose mob_atoms onto ref_atoms using SVD (Superimposer) and
    return (rmsd, n_atoms).
    """
    sup = Superimposer()
    sup.set_atoms(ref_atoms, mob_atoms)
    return round(sup.rms, 4), len(ref_atoms)


# ── Main loop ─────────────────────────────────────────────────────────────────

for pair in PAIRS:
    parent_dir = pair["parent_dir"]
    ref_path   = pair["reference"]
    label      = pair["label"]

    print(f"\n{'='*70}")
    print(f"Processing : {label}")
    print(f"  Reference : {ref_path}")
    print(f"  Models in : {parent_dir}")
    print(f"{'='*70}")

    if not os.path.isfile(ref_path):
        print(f"  WARNING: Reference not found — skipping.")
        continue

    model_paths = sorted(
        glob.glob(os.path.join(parent_dir, "**", "model.cif"), recursive=True)
    )
    if not model_paths:
        print(f"  WARNING: No model.cif files found — skipping.")
        continue

    print(f"  Found {len(model_paths)} model(s)")

    # Parse reference once
    ref_struct = PARSER.get_structure("reference", ref_path)
    ref_ca     = get_ca_atoms(ref_struct)
    ref_seq    = sequence_from_ca_list(ref_ca)
    print(f"  Reference: {len(ref_ca)} CA residues\n")

    results = []

    for model_path in model_paths:
        sample_name = os.path.basename(os.path.dirname(model_path))

        try:
            mob_struct = PARSER.get_structure("mobile", model_path)
            mob_ca     = get_ca_atoms(mob_struct)

            if not mob_ca:
                raise ValueError("No CA atoms found in mobile structure.")

            ref_atoms, mob_atoms = align_sequences_and_get_paired_atoms(ref_ca, mob_ca)

            if len(ref_atoms) < 10:
                raise ValueError(
                    f"Only {len(ref_atoms)} residues could be paired — "
                    "check sequence similarity."
                )

            rmsd, n_atoms = compute_rmsd(ref_atoms, mob_atoms)

            print(
                f"  {sample_name:<48s}"
                f"RMSD={rmsd:.4f} Å  "
                f"({n_atoms} paired CA atoms, "
                f"{len(mob_ca)} total in model)"
            )

            results.append({
                "sample":          sample_name,
                "rmsd_A":          rmsd,
                "n_atoms_paired":  n_atoms,
                "n_CA_model":      len(mob_ca),
                "n_CA_reference":  len(ref_ca),
                "model_path":      model_path,
            })

        except Exception as e:
            print(f"  ERROR — {sample_name}: {e}")
            results.append({
                "sample":         sample_name,
                "rmsd_A":         "ERROR",
                "n_atoms_paired": "ERROR",
                "n_CA_model":     "ERROR",
                "n_CA_reference": len(ref_ca),
                "model_path":     model_path,
            })

    # ── Averages ──────────────────────────────────────────────────────────────
    numeric  = [r["rmsd_A"] for r in results if isinstance(r["rmsd_A"], float)]
    avg_rmsd = round(sum(numeric) / len(numeric), 4) if numeric else "N/A"
    n_ok     = len(numeric)
    print(f"\n  Average RMSD: {avg_rmsd} Å  ({n_ok}/{len(results)} structures)")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path   = os.path.join(parent_dir, f"rmsd_results_{label}.csv")
    fieldnames = [
        "sample", "rmsd_A", "n_atoms_paired",
        "n_CA_model", "n_CA_reference", "model_path",
    ]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({f: "" for f in fieldnames})
        writer.writerow({
            "sample":         "AVERAGE",
            "rmsd_A":         avg_rmsd,
            "n_atoms_paired": "",
            "n_CA_model":     "",
            "n_CA_reference": "",
            "model_path":     f"{n_ok} structures",
        })

    print(f"  CSV written → {csv_path}")

print("\nDone.\n")
