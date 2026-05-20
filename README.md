# AlphaFold3 HPCC SLURM Workflow

## Overview

This repository contains SLURM batch scripts, input templates, CSV job files, helper utilities, and an environment file for running AlphaFold 3 jobs on the HPCC using Singularity.

The workflow is designed to support several common AlphaFold 3 run types:

1. Apo protein prediction
2. Protein-ligand cofolding
3. Templated apo protein prediction
4. Templated protein-ligand cofolding
5. Protein-protein cofolding

The repository is organized so that each workflow has a reusable JSON template, a CSV file describing job-specific inputs, and a corresponding SLURM batch script.

---

## Repository structure

```text
af3/
├── inputs/
│   ├── structures/
│   │   └── example_template_structure.cif
│   │
│   ├── af3_apo.csv
│   ├── af3_apo.json
│   ├── af3_cofold.csv
│   ├── af3_cofold.json
│   ├── af3_templated_apo.csv
│   ├── af3_templated_apo.json
│   ├── af3_templated_cofold.csv
│   └── af3_templated_cofold.json
│
├── logs/
│
├── outputs/
│
├── scripts/
│   ├── af3_apo.sb
│   ├── af3_cofold.sb
│   ├── af3_protprot_cofold.sb
│   ├── af3_templated_apo.sb
│   ├── af3_templated_cofold.sb
│   ├── cleanup.py
│   ├── pdb_to_cif.py
│   └── rmsd.py
│
├── tmp/
│
├── af3_environment.yaml
└── README.md
```

---

## Directory descriptions

### `inputs/`

The `inputs/` directory contains files used to generate AlphaFold 3 jobs.

This directory includes both reusable template files and job-specific files generated during runs.

Important reusable input files include:

```text
af3_apo.csv
af3_apo.json
af3_cofold.csv
af3_cofold.json
af3_templated_apo.csv
af3_templated_apo.json
af3_templated_cofold.csv
af3_templated_cofold.json
```

The `.csv` files define job-specific inputs. The `.json` files are reusable AlphaFold 3 JSON templates that the SLURM scripts copy and modify for each job.

Generated job-specific JSON files are also written into `inputs/`. These can be safely deleted after jobs are complete, but the reusable template JSON files should be kept.

---

### `inputs/structures/`

The `inputs/structures/` directory contains protein structure template files in `.cif` format.

Templated AlphaFold 3 workflows use this directory to locate template structures.

Example template files:

```text
inputs/structures/OATP1B1_inward.cif
inputs/structures/OATP1B1_outward.cif
inputs/structures/OATP1B3_inward.cif
inputs/structures/OATP1B3_outward.cif
```

Template CIF files should contain valid mmCIF-format structure data. The templated scripts are designed to automatically add required date metadata if it is missing from PyMOL-generated CIF files.

---

### `logs/`

The `logs/` directory contains SLURM output and error logs.

SLURM writes files here using names like:

```text
AF3_templated_cofold_123456_1.out
AF3_templated_cofold_123456_1.err
```

This directory can be cleaned with:

```bash
python scripts/cleanup.py
```

---

### `outputs/`

The `outputs/` directory contains AlphaFold 3 output directories.

Each job writes to its own output folder, usually named after the job tag generated from the CSV input.

Examples:

```text
outputs/OATP1B1/
outputs/OATP1B1_E3S/
outputs/ADK_ligand/
```

---

### `scripts/`

The `scripts/` directory contains SLURM batch scripts and helper utilities.

Important scripts include:

```text
af3_apo.sb
af3_cofold.sb
af3_protprot_cofold.sb
af3_templated_apo.sb
af3_templated_cofold.sb
cleanup.py
pdb_to_cif.py
rmsd.py
```

The `.sb` files are SLURM batch scripts. They can be submitted with `sbatch`.

Example:

```bash
sbatch scripts/af3_templated_cofold.sb
```

---

### `tmp/`

The `tmp/` directory is used as temporary working space during AlphaFold 3 runs.

Each job creates a temporary subdirectory inside `tmp/` and binds it into the Singularity container as a local temporary directory.

This directory can be cleaned with:

```bash
python scripts/cleanup.py
```

---

## Environment setup

A portable conda environment file is included:

```text
af3_environment.yaml
```

Create the environment with:

```bash
conda env create -f af3_environment.yaml
```

Activate it with:

```bash
conda activate af3
```

This environment is intended for helper scripts, structure processing, and analysis utilities. The AlphaFold 3 model itself is run inside the shared Singularity image specified in the SLURM scripts.

---

## Required external AlphaFold 3 resources

The SLURM scripts assume that shared AlphaFold 3 resources are available at:

```text
/mnt/research/woldring_lab/AlphaFold3
```

The scripts define this path using:

```bash
export AF3_RESOURCES_DIR=/mnt/research/woldring_lab/AlphaFold3
```

Expected resource structure:

```text
/mnt/research/woldring_lab/AlphaFold3/
├── image/
│   └── alphafold3.sif
├── code/
└── weights/
```

The AlphaFold 3 databases are expected at:

```text
/mnt/research/common-data/alphafold/database_3
```

These paths can be changed at the top of each `.sb` script if needed.

---

## Workflow overview

Each SLURM script follows the same general workflow:

1. Read the job row from the corresponding CSV file using `SLURM_ARRAY_TASK_ID`.
2. Load the reusable AlphaFold 3 JSON template.
3. Replace template fields with row-specific values.
4. Write a job-specific JSON file to `inputs/`.
5. Create a job-specific output directory in `outputs/`.
6. Create a job-specific temporary directory in `tmp/`.
7. Run AlphaFold 3 through the Singularity container.
8. Write logs to `logs/`.

---