#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import math
import numpy as np

try:
    import gemmi
except ImportError:
    print("ERROR: gemmi not found. Activate your env where gemmi is installed.", file=sys.stderr)
    sys.exit(1)


def read_structure_any(path: Path) -> gemmi.Structure:
    """
    Read a structure regardless of extension.
    If it looks like PDB, parse as PDB string. Otherwise let gemmi auto-detect.
    """
    text = path.read_text(errors="ignore")
    if text.lstrip().startswith(("ATOM", "HETATM", "MODEL", "HEADER", "REMARK", "CRYST1")):
        return gemmi.read_pdb_string(text)
    return gemmi.read_structure(str(path))


def kabsch(P: np.ndarray, Q: np.ndarray):
    """
    Return rotation R and translation t such that P@R + t aligns onto Q.
    P, Q: (N, 3)
    """
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc

    C = P0.T @ Q0
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Qc - Pc @ R
    return R, t


def apply_RT_to_model(model: gemmi.Model, R: np.ndarray, t: np.ndarray):
    """Apply x -> x@R + t to all atoms in the gemmi.Model."""
    for chain in model:
        for res in chain:
            for atom in res:
                p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                p2 = p @ R + t
                atom.pos = gemmi.Position(float(p2[0]), float(p2[1]), float(p2[2]))


def collect_ca_coords(model: gemmi.Model):
    """
    Collect CA coordinates keyed by (chain_id, resseq, icode).
    Only keeps polymer ATOM residues (het_flag == 'A'), skips waters/ligands.
    Returns dict[key] = (coord(3,), sort_key)
    """
    out = {}
    for chain in model:
        cid = chain.name
        for res in chain:
            if getattr(res, "het_flag", None) != "A":
                continue
            ca = None
            for atom in res:
                if atom.name == "CA":
                    ca = atom
                    break
            if ca is None:
                continue

            resseq = res.seqid.num
            icode = res.seqid.icode if res.seqid.icode else ""
            key = (cid, resseq, icode)
            coord = np.array([ca.pos.x, ca.pos.y, ca.pos.z], dtype=float)
            sort_key = (cid, resseq, icode)
            out[key] = (coord, sort_key)
    return out


def intersection_coords(ref_ca: dict, mob_ca: dict):
    """Return matched coordinate arrays (mob, ref) for common CA keys, ordered by ref sort_key."""
    common = sorted(set(ref_ca.keys()) & set(mob_ca.keys()), key=lambda k: ref_ca[k][1])
    if not common:
        return None, None, None
    P = np.stack([mob_ca[k][0] for k in common], axis=0)
    Q = np.stack([ref_ca[k][0] for k in common], axis=0)
    sort_keys = [ref_ca[k][1] for k in common]
    return P, Q, sort_keys


def rmsd(P: np.ndarray, Q: np.ndarray):
    d2 = np.sum((P - Q) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)))


def membrane_axis_from_pca(coords: np.ndarray):
    """
    Estimate membrane normal as the PCA axis with smallest variance.
    Returns unit vector n and center c.
    """
    c = coords.mean(axis=0)
    X = coords - c
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    evals, evecs = np.linalg.eigh(C)  # ascending
    n = evecs[:, 0]                   # smallest variance axis
    n = n / np.linalg.norm(n)
    return n, c


def min_interdomain_distance(P: np.ndarray, sort_keys, s_vals: np.ndarray, s_lo: float, s_hi: float):
    """
    Gate distance = minimum CA-CA distance between N-half and C-half residues
    within a membrane-coordinate slice (s_lo < s < s_hi).

    Split halves by residue order in sort_keys.
    """
    in_slice = (s_vals > s_lo) & (s_vals < s_hi)
    idx = np.where(in_slice)[0]
    if idx.size < 20:
        return float("nan")

    n = len(sort_keys)
    mid = n // 2
    n_half_mask = (np.arange(n) < mid)
    c_half_mask = ~n_half_mask

    A_idx = idx[n_half_mask[idx]]
    B_idx = idx[c_half_mask[idx]]

    if A_idx.size < 10 or B_idx.size < 10:
        return float("nan")

    A = P[A_idx, :]
    B = P[B_idx, :]

    diff = A[:, None, :] - B[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(dist2)))


def model_to_single_pdb(structure_template: gemmi.Structure, model: gemmi.Model, out_pdb: Path):
    """
    Write a single model to PDB as a standalone structure.
    """
    st = gemmi.Structure()
    st.name = structure_template.name
    st.spacegroup_hm = structure_template.spacegroup_hm
    st.cell = structure_template.cell

    m = gemmi.Model("1")
    for chain in model:
        new_chain = gemmi.Chain(chain.name)
        for res in chain:
            new_res = gemmi.Residue()
            new_res.name = res.name
            new_res.seqid = gemmi.SeqId(res.seqid.num, res.seqid.icode)
            new_res.het_flag = res.het_flag
            for atom in res:
                new_atom = gemmi.Atom()
                new_atom.name = atom.name
                new_atom.altloc = atom.altloc
                new_atom.occ = atom.occ
                new_atom.b_iso = atom.b_iso
                new_atom.element = atom.element
                new_atom.charge = atom.charge
                new_atom.pos = gemmi.Position(atom.pos.x, atom.pos.y, atom.pos.z)
                new_res.add_atom(new_atom)
            new_chain.add_residue(new_res)
        m.add_chain(new_chain)

    st.add_model(m)
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb.write_text(st.make_pdb_string())


def main():
    ap = argparse.ArgumentParser(
        description="Compare AF3 ensemble states to OATP1B1 apo using auto-TM RMSD + PCA-normal gate distances; export top models."
    )
    ap.add_argument("--ref", required=True, type=Path, help="Reference apo structure (PDB ok)")
    ap.add_argument("--ens", required=True, type=Path, help="Ensemble multi-model PDB (100 states)")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--top", type=int, default=5, help="How many to export (default 5)")
    ap.add_argument("--zcut", type=float, default=18.0, help="TM slab half-thickness along PCA normal in Å (default 18)")
    ap.add_argument("--slice", type=float, default=6.0, help="Gate slice thickness at each slab end in Å (default 6)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite exported PDBs if present")
    args = ap.parse_args()

    ref_path = args.ref.resolve()
    ens_path = args.ens.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ref_st = read_structure_any(ref_path)
    ens_st = read_structure_any(ens_path)

    if len(ref_st) < 1:
        raise RuntimeError("Reference has no models")
    if len(ens_st) < 1:
        raise RuntimeError("Ensemble has no models")

    ref_model = ref_st[0]
    ref_ca = collect_ca_coords(ref_model)
    if len(ref_ca) < 50:
        raise RuntimeError("Too few CA atoms in reference (did parsing work?)")

    # Build ordered reference arrays for PCA
    ref_items = sorted(ref_ca.items(), key=lambda kv: kv[1][1])
    Q_ref_all = np.stack([kv[1][0] for kv in ref_items], axis=0)
    ref_sort_keys_all = [kv[1][1] for kv in ref_items]

    # PCA membrane axis (normal) and center
    n_axis, center = membrane_axis_from_pca(Q_ref_all)
    s_ref_all = (Q_ref_all - center) @ n_axis
    s_mid = float(np.median(s_ref_all))

    # Define slab and gate slices in membrane coordinate s
    slab_lo = s_mid - args.zcut
    slab_hi = s_mid + args.zcut

    top_lo = s_mid + (args.zcut - args.slice)
    top_hi = s_mid + args.zcut
    bot_lo = s_mid - args.zcut
    bot_hi = s_mid - (args.zcut - args.slice)

    # Reference gates computed on reference itself (may still be NaN; that's ok)
    tm_ref_mask = (s_ref_all > slab_lo) & (s_ref_all < slab_hi)
    ref_ext_gate = min_interdomain_distance(Q_ref_all[tm_ref_mask],
                                           [k for k, m in zip(ref_sort_keys_all, tm_ref_mask) if m],
                                           s_ref_all[tm_ref_mask], top_lo, top_hi)
    ref_int_gate = min_interdomain_distance(Q_ref_all[tm_ref_mask],
                                           [k for k, m in zip(ref_sort_keys_all, tm_ref_mask) if m],
                                           s_ref_all[tm_ref_mask], bot_lo, bot_hi)

    results = []

    # Iterate ensemble states
    for state_idx, mob_model in enumerate(ens_st, start=1):
        mob_ca = collect_ca_coords(mob_model)
        P_all, Q_all, sort_keys = intersection_coords(ref_ca, mob_ca)
        if P_all is None or P_all.shape[0] < 80:
            continue

        # Membrane-coordinate on reference matched set (define TM mask by ref coords)
        s_match_ref = (Q_all - center) @ n_axis
        tm_mask = (s_match_ref > slab_lo) & (s_match_ref < slab_hi)
        if tm_mask.sum() < 40:
            continue

        P_tm = P_all[tm_mask]
        Q_tm = Q_all[tm_mask]
        sort_keys_tm = [k for k, keep in zip(sort_keys, tm_mask) if keep]

        # Fit TM -> TM
        R, t = kabsch(P_tm, Q_tm)
        P_tm_aligned = P_tm @ R + t

        tm_r = rmsd(P_tm_aligned, Q_tm)

        # Gate distances on ALIGNED TM coords, using membrane coordinate s on aligned coords
        s_tm_aligned = (P_tm_aligned - center) @ n_axis
        ext_gate = min_interdomain_distance(P_tm_aligned, sort_keys_tm, s_tm_aligned, top_lo, top_hi)
        int_gate = min_interdomain_distance(P_tm_aligned, sort_keys_tm, s_tm_aligned, bot_lo, bot_hi)

        # Outward score: prefer larger ext opening and smaller int opening.
        # If reference gates are NaN, fall back to ext - int (still useful).
        if math.isnan(ext_gate) or math.isnan(int_gate):
            outward_score = float("nan")
        else:
            if math.isnan(ref_ext_gate) or math.isnan(ref_int_gate):
                outward_score = ext_gate - int_gate
            else:
                outward_score = (ext_gate - ref_ext_gate) - (int_gate - ref_int_gate)

        results.append({
            "state": state_idx,
            "tm_rmsd_ca": tm_r,
            "ext_gate_min_ca": ext_gate,
            "int_gate_min_ca": int_gate,
            "outward_score": outward_score
        })

    if not results:
        raise RuntimeError("No valid models processed. Check inputs / CA overlap.")

    # Write CSV
    import csv
    metrics_fp = outdir / "tm_rmsd_gate_metrics.csv"
    with metrics_fp.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["state", "tm_rmsd_ca", "ext_gate_min_ca", "int_gate_min_ca", "outward_score"]
        )
        w.writeheader()
        for r in sorted(results, key=lambda d: d["state"]):
            w.writerow(r)

    # Top by TM RMSD (lowest)
    by_rmsd = sorted(results, key=lambda d: d["tm_rmsd_ca"])
    top_rmsd = by_rmsd[:args.top]

    # Top by outward score (highest), ignoring NaNs
    by_out = [r for r in results if not math.isnan(r["outward_score"])]
    by_out.sort(key=lambda d: d["outward_score"], reverse=True)
    top_out = by_out[:args.top]

    # Export aligned full models for chosen states
    top_rmsd_dir = outdir / "top_by_tm_rmsd"
    top_out_dir = outdir / "top_by_outward_score"
    top_rmsd_dir.mkdir(parents=True, exist_ok=True)
    top_out_dir.mkdir(parents=True, exist_ok=True)

    def export_states(chosen, target_dir, tag):
        for rank, r in enumerate(chosen, start=1):
            state = r["state"]
            out_pdb = target_dir / (
                f"{tag}_rank{rank:02d}_state{state:03d}"
                f"_tmRMSD{r['tm_rmsd_ca']:.3f}"
                f"_ext{(r['ext_gate_min_ca'] if not math.isnan(r['ext_gate_min_ca']) else -1):.2f}"
                f"_int{(r['int_gate_min_ca'] if not math.isnan(r['int_gate_min_ca']) else -1):.2f}.pdb"
            )
            if out_pdb.exists() and not args.overwrite:
                continue

            mob_model = ens_st[state - 1]
            mob_ca = collect_ca_coords(mob_model)
            P_all, Q_all, sort_keys = intersection_coords(ref_ca, mob_ca)
            if P_all is None:
                continue

            s_match_ref = (Q_all - center) @ n_axis
            tm_mask = (s_match_ref > slab_lo) & (s_match_ref < slab_hi)
            if tm_mask.sum() < 40:
                continue

            P_tm = P_all[tm_mask]
            Q_tm = Q_all[tm_mask]

            R, t = kabsch(P_tm, Q_tm)

            # Apply transform to FULL model and write
            apply_RT_to_model(mob_model, R, t)
            model_to_single_pdb(ens_st, mob_model, out_pdb)

    export_states(top_rmsd, top_rmsd_dir, "TM_RMSD")
    export_states(top_out, top_out_dir, "OUTWARD")

    # Summary
    print(f"Wrote metrics: {metrics_fp}")
    print(f"Membrane axis (PCA smallest-variance) n={n_axis}")
    print(f"s_mid={s_mid:.2f} Å, zcut=±{args.zcut:.1f} Å, slice={args.slice:.1f} Å")
    print(f"Reference gates: ext={ref_ext_gate:.2f} Å, int={ref_int_gate:.2f} Å")
    print(f"Exported top {len(top_rmsd)} by TM RMSD -> {top_rmsd_dir}")
    print(f"Exported top {len(top_out)} by outward score -> {top_out_dir}")


if __name__ == "__main__":
    main()
