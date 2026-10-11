"""CLI script to download and verify the Semiconductor Wafer Etch dataset (Bobick & Olszewski 2001)."""

from __future__ import annotations

import argparse
import json
import sys

from yieldguard.wafer_data import download_and_prepare_wafer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and prepare the Semiconductor Wafer Etch dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/wafer", help="Raw download directory.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/wafer_runs.parquet",
        help="Processed parquet path.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/wafer_source_manifest.json",
        help="Output source manifest JSON path.",
    )
    args = parser.parse_args(argv)

    manifest = download_and_prepare_wafer(
        raw_dir=args.raw_dir,
        processed_path=args.output,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
