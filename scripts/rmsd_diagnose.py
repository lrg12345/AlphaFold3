#!/usr/bin/env python3
"""
DIAGNOSTIC SCRIPT — run this first before rmsd.py
Loads one reference CIF and one model CIF and prints exactly what PyMOL sees:
  - chains, residue count, atom count, coordinate range
This will reveal why alignment is failing.

Usage:
    pymol -c /mnt/gs21/scratch/garlan70/af3/scripts/rmsd_diagnose.py
"""

import sys

try:
    import pymol
    from pymol import cmd
    pymol.finish_launching(["pymol", "-qc"])
except ImportError:
    sys.exit("ERROR: PyMOL not found.")

# ── Edit these two paths to a known reference + a known bad model ─────────────
REF_PATH    = "/mnt/gs21/scratch/garlan70/af3/inputs/structures/OATP1B1_inward.cif"
MOBILE_PATH = "/mnt/gs21/scratch/garlan70/af3/outputs/OATP1B1_inward/oatp1b1_inward/seed-1798559295_sample-3/model.cif"
# ─────────────────────────────────────────────────────────────────────────────

def diagnose(name, path):
    print(f"\n{'─'*60}")
    print(f"Object : {name}")
    print(f"File   : {path}")
    cmd.delete(name)
    cmd.load(path, name)

    total_atoms = cmd.count_atoms(name)
    print(f"Total atoms loaded : {total_atoms}")

    # States (AF3 CIFs sometimes load as multi-state)
    n_states = cmd.count_states(name)
    print(f"Number of states   : {n_states}  ← should be 1; if >1, AF3 packed multiple models")

    # Chains
    chains = cmd.get_chains(name)
    print(f"Chains             : {chains}")

    # Per-chain breakdown
    for ch in chains:
        sel = f"{name} and chain {ch}"
        n_atoms  = cmd.count_atoms(sel)
        n_ca     = cmd.count_atoms(f"{sel} and name CA")
        n_res    = cmd.count_atoms(f"{sel} and name CA and polymer")
        print(f"  Chain {ch}: {n_atoms} atoms, {n_ca} CA atoms, ~{n_res} residues")

    # Coordinate range — huge values = structure at wrong origin
    coords = cmd.get_coords(name)
    if coords is not None and len(coords) > 0:
        import numpy as np
        print(f"  X range: {coords[:,0].min():.1f} to {coords[:,0].max():.1f}")
        print(f"  Y range: {coords[:,1].min():.1f} to {coords[:,1].max():.1f}")
        print(f"  Z range: {coords[:,2].min():.1f} to {coords[:,2].max():.1f}")
        centroid = coords.mean(axis=0)
        print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")
    else:
        print("  WARNING: Could not retrieve coordinates!")

    # First 5 residues to check numbering
    space = {"residues": []}
    cmd.iterate(
        f"{name} and name CA and polymer",
        "residues.append((chain, resi, resn))",
        space=space,
    )
    res_list = space["residues"]
    print(f"  First 5 residues : {res_list[:5]}")
    print(f"  Last  5 residues : {res_list[-5:]}")
    print(f"  Total CA residues: {len(res_list)}")


diagnose("reference", REF_PATH)
diagnose("mobile",    MOBILE_PATH)

# ── Attempt super and report ──────────────────────────────────────────────────
print(f"\n{'─'*60}")
print("Attempting cmd.super('mobile', 'reference') ...")
try:
    result = cmd.super("mobile", "reference")
    print(f"  super() returned: RMSD={result[0]:.4f} Å, n_atoms={result[1]}")
    if result[0] > 10:
        print("  !! RMSD still high after super — likely a chain/state mismatch above.")
    else:
        print("  Alignment looks good.")
except Exception as e:
    print(f"  super() FAILED: {e}")

# ── Try restricting to chain A explicitly ────────────────────────────────────
print("\nAttempting super on 'chain A and polymer' only ...")
try:
    result2 = cmd.super(
        "mobile and chain A and polymer",
        "reference and chain A and polymer",
    )
    print(f"  super(chain A) returned: RMSD={result2[0]:.4f} Å, n_atoms={result2[1]}")
except Exception as e:
    print(f"  super(chain A) FAILED: {e}")

print("\nDiagnostic complete.\n")
cmd.quit()