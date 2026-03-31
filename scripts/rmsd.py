#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.SVDSuperimposer import SVDSuperimposer


ResidueKey = Tuple[str, int, str]


def is_standard_protein_residue(res: Residue) -> bool:
    return res.id[0] == " " and "CA" in res


def residue_key(chain: Chain, res: Residue) -> ResidueKey:
    _, resseq, icode = res.id
    return chain.id, int(resseq), str(icode).strip()


def load_structure(cif_path: Path, struct_id: str):
    parser = MMCIFParser(QUIET=True)
    return parser.get_structure(struct_id, str(cif_path))


def get_first_model(structure) -> Model:
    return next(structure.get_models())


def collect_ca_atoms(model: Model, chain_id: str) -> Dict[ResidueKey, Atom]:
    out: Dict[ResidueKey, Atom] = {}
    for chain in model:
        if chain.id != chain_id:
            continue
        for res in chain:
            if not is_standard_protein_residue(res):
                continue
            out[residue_key(chain, res)] = res["CA"]
    return out


def matched_atoms(
    ref_model: Model,
    mob_model: Model,
    chain_id: str,
) -> Tuple[List[ResidueKey], np.ndarray, np.ndarray]:
    ref_atoms = collect_ca_atoms(ref_model, chain_id)
    mob_atoms = collect_ca_atoms(mob_model, chain_id)

    keys = sorted(set(ref_atoms) & set(mob_atoms), key=lambda x: (x[1], x[2]))
    if not keys:
        raise ValueError("No matched CA residues found")

    ref_xyz = np.array([ref_atoms[k].coord for k in keys], dtype=float)
    mob_xyz = np.array([mob_atoms[k].coord for k in keys], dtype=float)
    return keys, ref_xyz, mob_xyz


def fit_transform(ref_xyz: np.ndarray, mob_xyz: np.ndarray):
    sup = SVDSuperimposer()
    sup.set(ref_xyz, mob_xyz)
    sup.run()
    rot, tran = sup.get_rotran()
    rms = float(sup.get_rms())
    return rot, tran, rms


def apply_transform(xyz: np.ndarray, rot: np.ndarray, tran: np.ndarray) -> np.ndarray:
    return np.dot(xyz, rot) + tran


def per_atom_distances(ref_xyz: np.ndarray, mob_xyz: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((ref_xyz - mob_xyz) ** 2, axis=1))


def rmsd(ref_xyz: np.ndarray, mob_xyz: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((ref_xyz - mob_xyz) ** 2, axis=1))))


def iterative_prune_fit(
    ref_xyz: np.ndarray,
    mob_xyz: np.ndarray,
    cutoff: float = 2.0,
    max_cycles: int = 5,
    min_atoms: int = 20,
):
    keep = np.ones(len(ref_xyz), dtype=bool)

    for _ in range(max_cycles):
        if keep.sum() < max(min_atoms, 3):
            break

        rot, tran, _ = fit_transform(ref_xyz[keep], mob_xyz[keep])
        mob_fit_all = apply_transform(mob_xyz, rot, tran)
        dists = per_atom_distances(ref_xyz, mob_fit_all)

        new_keep = dists <= cutoff
        if new_keep.sum() < max(min_atoms, 3):
            break
        if np.array_equal(new_keep, keep):
            keep = new_keep
            break
        keep = new_keep

    rot, tran, _ = fit_transform(ref_xyz[keep], mob_xyz[keep])
    mob_fit_all = apply_transform(mob_xyz, rot, tran)
    final_rms = rmsd(ref_xyz[keep], mob_fit_all[keep])
    return final_rms, int(keep.sum()), keep


def find_models(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("model.cif") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compute pruned CA RMSDs for AF3 model.cif files against a reference CIF."
    )
    ap.add_argument(
        "--ref",
        type=Path,
        default=Path("/mnt/gs21/scratch/garlan70/af3/inputs/structures/ADK_closed.cif"), # CHANGE ME
        help="Reference CIF path",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/gs21/scratch/garlan70/af3/outputs/ADK/ADK_ATP_seeded_closed"), # CHANGE ME
        help="Root directory to search recursively for model.cif files",
    )
    ap.add_argument("--chain", default="A", help="Protein chain ID")
    ap.add_argument(
        "--cutoff",
        type=float,
        default=2.0,
        help="Pruning cutoff in Å after each fit cycle",
    )
    ap.add_argument(
        "--cycles",
        type=int,
        default=5,
        help="Maximum number of prune/refit cycles",
    )
    ap.add_argument(
        "--min-atoms",
        type=int,
        default=20,
        help="Minimum number of CA atoms to retain during pruning",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = ap.parse_args()

    ref_struct = load_structure(args.ref, "ref")
    ref_model = get_first_model(ref_struct)

    model_paths = find_models(args.root)
    if not model_paths:
        raise SystemExit(f"No model.cif files found under {args.root}")

    rows = []
    failures = []

    # self-check
    keys_self, ref_self, mob_self = matched_atoms(ref_model, ref_model, args.chain)
    self_rms, self_n, _ = iterative_prune_fit(
        ref_self,
        mob_self,
        cutoff=args.cutoff,
        max_cycles=args.cycles,
        min_atoms=args.min_atoms,
    )
    print(f"[self-check:pruned] matched={self_n}, RMSD={self_rms:.6f} Å")

    for path in model_paths:
        try:
            mob_struct = load_structure(path, path.stem)
            mob_model = get_first_model(mob_struct)

            keys, ref_xyz, mob_xyz = matched_atoms(ref_model, mob_model, args.chain)

            val, used, _ = iterative_prune_fit(
                ref_xyz,
                mob_xyz,
                cutoff=args.cutoff,
                max_cycles=args.cycles,
                min_atoms=args.min_atoms,
            )

            rows.append((str(path), val, len(keys), used))
        except Exception as e:
            failures.append((str(path), str(e)))

    if not rows:
        raise SystemExit("All model evaluations failed")

    rows.sort(key=lambda x: x[1])
    avg = sum(r[1] for r in rows) / len(rows)
    lo = rows[0]
    hi = rows[-1]

    print(f"Reference CIF: {args.ref}")
    print(f"Search root:    {args.root}")
    print("Mode:           pruned")
    print(f"Cutoff:         {args.cutoff} Å")
    print(f"Cycles:         {args.cycles}")
    print(f"Matched models: {len(rows)}")
    print(f"Failures:       {len(failures)}")
    print()
    print(f"Average RMSD: {avg:.4f} Å")
    print(f"Lowest RMSD:  {lo[1]:.4f} Å")
    print(f"  File:       {lo[0]}")
    print(f"  Matched CA: {lo[2]}")
    print(f"  Used CA:    {lo[3]}")
    print(f"Highest RMSD: {hi[1]:.4f} Å")
    print(f"  File:       {hi[0]}")
    print(f"  Matched CA: {hi[2]}")
    print(f"  Used CA:    {hi[3]}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "rmsd_angstrom", "matched_ca", "used_ca"])
            for row in rows:
                w.writerow([row[0], f"{row[1]:.6f}", row[2], row[3]])

    if failures:
        print("\nFailures:")
        for p, msg in failures:
            print(f"  {p}\n    {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())