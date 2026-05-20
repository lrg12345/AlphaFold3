#!/usr/bin/env python3

from pathlib import Path

LOG_DIR = Path("/mnt/gs21/scratch/garlan70/af3/logs")
INPUT_DIR = Path("/mnt/gs21/scratch/garlan70/af3/inputs")
TMP_DIR = Path("/mnt/gs21/scratch/garlan70/af3/tmp")

PROTECTED_INPUT_JSONS = {
    "af3_apo.json",
    "af3_cofold.json",
    "af3_templated_apo.json",
    "af3_templated_cofold.json",
}


def delete_files_in_directory(directory: Path, pattern: str = "*") -> None:
    if not directory.exists():
        print(f"Skipping missing directory: {directory}")
        return

    if not directory.is_dir():
        print(f"Skipping non-directory path: {directory}")
        return

    deleted_count = 0

    for path in directory.glob(pattern):
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted_count += 1

    print(f"Deleted {deleted_count} file(s) from {directory}")


def delete_job_jsons(directory: Path) -> None:
    if not directory.exists():
        print(f"Skipping missing directory: {directory}")
        return

    if not directory.is_dir():
        print(f"Skipping non-directory path: {directory}")
        return

    deleted_count = 0
    skipped_count = 0

    for path in directory.glob("*.json"):
        if path.name in PROTECTED_INPUT_JSONS:
            skipped_count += 1
            continue

        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted_count += 1

    print(f"Deleted {deleted_count} job-specific JSON file(s) from {directory}")
    print(f"Protected {skipped_count} template JSON file(s) from {directory}")


def main() -> None:
    print("Starting AF3 cleanup...")

    # Delete all log files
    delete_files_in_directory(LOG_DIR)

    # Delete job-specific JSON input files, but keep reusable template JSONs
    delete_job_jsons(INPUT_DIR)

    # Delete all job-specific temporary files
    delete_files_in_directory(TMP_DIR)

    print("Cleanup complete.")


if __name__ == "__main__":
    main()