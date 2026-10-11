"""Download and extract UCI SECOM dataset with provenance, Zip Slip guard, and SHA-256 verification."""

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
DATASET_PAGE_URL = "https://archive.ics.uci.edu/dataset/179/secom"
DOI = "10.24432/C54305"
LICENCE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
CITATION = (
    "McCann, M. & Johnston, A. (2008). SECOM [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C54305."
)

EXPECTED_HASHES = {
    "archive": "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e",
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
    "secom.names": "6d91b0b46cdee03064ee3e3112f937c1b3f7fcd9933575794ec07974e6f1ea59",
}
REQUIRED_FILES = ["secom.data", "secom_labels.data", "secom.names"]


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_extract_zip(
    zip_bytes: bytes, target_dir: Path, allowed_filenames: list[str]
) -> dict[str, str]:
    """Safely extract specified files from zip bytes preventing directory traversal (Zip Slip)."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_hashes: dict[str, str] = {}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for member in z.namelist():
            if ".." in Path(member).parts or member.startswith(("/", "\\")) or ":" in member:
                raise ValueError(
                    f"Security error: Zip member {member} attempts path traversal outside target directory."
                )
            resolved_member_path = (target_dir / member).resolve()
            if not resolved_member_path.is_relative_to(target_dir):
                raise ValueError(
                    f"Security error: Zip member {member} attempts path traversal outside target directory."
                )

            filename = os.path.basename(member)
            if filename in allowed_filenames:
                dest_path = (target_dir / filename).resolve()
                if not dest_path.is_relative_to(target_dir):
                    raise ValueError(
                        f"Security error: Destination {dest_path} is outside target directory."
                    )

                content = z.read(member)
                dest_path.write_bytes(content)
                extracted_hashes[filename] = compute_sha256(content)

    return extracted_hashes


def download_and_extract_secom(output_dir: Path, manifest_path: Path) -> dict:
    """Download UCI SECOM archive, verify hashes, extract allowed files, and write source manifest."""
    project_root = Path.cwd().resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading UCI SECOM dataset from {DATASET_URL} ...")
    response = requests.get(DATASET_URL, timeout=60)
    response.raise_for_status()
    archive_bytes = response.content
    archive_hash = compute_sha256(archive_bytes)
    print(f"Archive downloaded ({len(archive_bytes)} bytes, SHA-256: {archive_hash})")

    extracted_hashes = safe_extract_zip(archive_bytes, output_dir, REQUIRED_FILES)
    print(f"Extracted {len(extracted_hashes)} files to {output_dir}")

    missing = [f for f in REQUIRED_FILES if f not in extracted_hashes]
    if missing:
        raise ValueError(f"Missing required extracted files: {missing}")

    hash_checks = {}
    mismatches = []
    for fname, expected in EXPECTED_HASHES.items():
        actual = archive_hash if fname == "archive" else extracted_hashes.get(fname)
        matches = actual == expected
        hash_checks[fname] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            mismatches.append(f"{fname}: expected {expected}, got {actual}")

    if mismatches:
        raise ValueError(f"SHA-256 checksum mismatch for SECOM archive/files: {'; '.join(mismatches)}")

    files_meta = {}
    for fname in REQUIRED_FILES:
        fpath = output_dir / fname
        try:
            rel_path = str(fpath.relative_to(project_root).as_posix())
        except ValueError:
            rel_path = str(fpath)
        files_meta[fname] = {
            "sha256": extracted_hashes[fname],
            "size_bytes": fpath.stat().st_size,
            "path": rel_path,
        }

    manifest_data = {
        "dataset_id": "SECOM",
        "source_url": DATASET_URL,
        "dataset_page_url": DATASET_PAGE_URL,
        "doi": DOI,
        "licence": LICENCE,
        "licence_verified_date": "2026-10-11",
        "citation": CITATION,
        "download_timestamp_utc": datetime.now(UTC).isoformat(),
        "archive_sha256": archive_hash,
        "files": files_meta,
        "hash_checks": hash_checks,
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"Saved source manifest to {manifest_path}")
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and extract UCI SECOM dataset.")
    parser.add_argument("--dataset", type=str, default="SECOM", help="Dataset identifier (SECOM only)")
    parser.add_argument("--output", type=str, default="data/raw/secom", help="Target output directory")
    parser.add_argument(
        "--manifest", type=str, default="data/source_manifest.json", help="Manifest output path"
    )
    args = parser.parse_args()

    if args.dataset.upper() != "SECOM":
        print(f"Error: Dataset {args.dataset} is outside MVP scope (SECOM only).", file=sys.stderr)
        sys.exit(1)

    download_and_extract_secom(Path(args.output), Path(args.manifest))


if __name__ == "__main__":
    main()
