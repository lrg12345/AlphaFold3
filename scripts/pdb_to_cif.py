#!/usr/bin/env python3
import re
import gemmi
from pathlib import Path

BASE_DIR = Path("/mnt/gs21/scratch/garlan70/af3/inputs/structures")
PDB_NAME = "OATP1B3"

INPUT_PDB = BASE_DIR / f"{PDB_NAME}.pdb"
OUTPUT_CIF = BASE_DIR / f"{PDB_NAME}.cif"

if not INPUT_PDB.exists():
    raise SystemExit(f"ERROR: {INPUT_PDB} not found")

print("==> Reading PDB with gemmi")
st = gemmi.read_structure(str(INPUT_PDB))
model = st[0]

# ---- Keep only one chain ----
chain_info = []
for ch in model:
    n_res = len(ch)
    n_atoms = sum(len(res) for res in ch)
    chain_info.append((ch.name, n_res, n_atoms))

print("==> Chains found (chain_id, residues, atoms):")
for name, n_res, n_atoms in chain_info:
    print(f"    '{name}': residues={n_res}, atoms={n_atoms}")

if not chain_info:
    raise SystemExit("ERROR: No chains found in model 0")

chain_names = [x[0] for x in chain_info]
if "A" in chain_names:
    keep_name = "A"
else:
    keep_name = max(chain_info, key=lambda t: t[2])[0]
    print(f"==> WARNING: chain 'A' not found; keeping largest chain '{keep_name}'")

for ch in list(model):
    if ch.name != keep_name:
        try:
            del model[ch.name]
        except Exception:
            model.remove_chain(ch.name)

# Rename remaining chain to 'A'
for ch in model:
    ch.name = "A"

st.setup_entities()

# Sanity check
atom_count = sum(len(res) for ch in model for res in ch)
if atom_count == 0:
    raise SystemExit("ERROR: No atoms retained")

print(f"==> Atoms retained: {atom_count}")
print("==> Writing mmCIF")
st.make_mmcif_document().write_file(str(OUTPUT_CIF))

# ============================================================
# Post-fix for AF3 compatibility
# ============================================================

print("==> Post-fixing CIF for AF3")

text = OUTPUT_CIF.read_text()

# Fix chain tokens like Axp -> A
text = re.sub(r'(?<=\s)Axp(?=\s)', 'A', text)

lines = text.splitlines(True)

in_atom_loop = False
headers = []
col_map = {}
output_lines = []

i = 0
while i < len(lines):
    line = lines[i]

    # Detect atom_site loop
    if line.startswith("loop_"):
        in_atom_loop = False
        headers = []
        col_map = {}
        output_lines.append(line)
        i += 1
        continue

    if line.startswith("_atom_site."):
        in_atom_loop = True
        headers.append(line.strip())
        output_lines.append(line)
        i += 1
        continue

    # Build column map once headers are done
    if in_atom_loop and headers and not line.startswith("_atom_site."):
        col_map = {h: idx for idx, h in enumerate(headers)}
        in_atom_loop = False  # only build once

    # Fix ATOM lines
    if line.startswith("ATOM"):
        parts = line.split()

        if "_atom_site.label_seq_id" not in col_map or "_atom_site.auth_seq_id" not in col_map:
            raise SystemExit("ERROR: Required atom_site columns missing")

        i_label = col_map["_atom_site.label_seq_id"]
        i_auth = col_map["_atom_site.auth_seq_id"]

        auth_id = int(parts[i_auth])
        parts[i_label] = str(auth_id + 1)

        output_lines.append(" ".join(parts) + "\n")
        i += 1
        continue

    output_lines.append(line)
    i += 1

fixed_text = "".join(output_lines)

# --- Ensure AF3-required "release date" metadata exists ---
# AF3 template parsing requires a release/deposition date in the mmCIF.
# If missing, inject minimal fields right after _entry.id.
if "_pdbx_database_status.recvd_initial_deposition_date" not in fixed_text and \
   "_pdbx_audit_revision_history.revision_date" not in fixed_text:

    inject = (
        "\n"
        "_pdbx_database_status.recvd_initial_deposition_date 2020-01-01\n"
        "loop_\n"
        "_pdbx_audit_revision_history.revision_date\n"
        "2020-01-01\n"
    )

    # Insert right after the _entry.id line (or at top if not found)
    m = re.search(r"(?m)^_entry\.id.*$", fixed_text)
    if m:
        insert_pos = m.end()
        fixed_text = fixed_text[:insert_pos] + inject + fixed_text[insert_pos:]
    else:
        fixed_text = inject + fixed_text

OUTPUT_CIF.write_text(fixed_text)
print(f"OK: wrote AF3-compatible CIF -> {OUTPUT_CIF}")
