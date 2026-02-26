#!/usr/bin/env python3
"""Split gemm_sizes_by_chip.csv into Small/Medium/Large CSVs with snippets.

Reads ~/temp/gemm_sizes_by_chip.csv and outputs 6 files into SMLGemm/:
  SmallGemm.csv, SmallGemmSnippet.csv,
  MediumGemm.csv, MediumGemmSnippet.csv,
  LargeGemm.csv, LargeGemmSnippet.csv

Usage:
  python dump_sml_gemm.py                          # default chip: mi300x
  python dump_sml_gemm.py --chip rx9070xt
  python dump_sml_gemm.py --chip mi350x --snippet-count 32
  python dump_sml_gemm.py --input /path/to/gemm_sizes_by_chip.csv
"""

import argparse
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path.home() / "temp" / "gemm_sizes_by_chip.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "SMLGemm"
DEFAULT_CHIP = "mi300x"
DEFAULT_SNIPPET_COUNT = 16

CATEGORIES = ["SmallGemm", "MediumGemm", "LargeGemm"]
OUTPUT_FIELDS = ["filename", "M", "N", "K", "batch", "tuning_folder_path", "proxy_shapes"]


def load_gemm_csv(path: Path) -> tuple[list[dict], list[str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    chip_cols = [c for c in (reader.fieldnames or []) if c.startswith("gemm size of ")]
    return rows, chip_cols


def filter_by_category(rows: list[dict], chip_col: str, category: str) -> list[dict]:
    return [r for r in rows if r.get(chip_col, "").strip() == category]


def to_output_row(row: dict) -> dict:
    return {
        "filename": row["filename"],
        "M": row["M"],
        "N": row["N"],
        "K": row["K"],
        "batch": row["batch"],
        "tuning_folder_path": "",
        "proxy_shapes": row.get("proxy_shapes", ""),
    }


def write_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to gemm_sizes_by_chip.csv")
    parser.add_argument("--chip", default=DEFAULT_CHIP, help=f"Chip to filter by (default: {DEFAULT_CHIP})")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--snippet-count", type=int, default=DEFAULT_SNIPPET_COUNT, help="Number of rows in snippet files")
    args = parser.parse_args()

    rows, chip_cols = load_gemm_csv(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Available chips: {[c.removeprefix('gemm size of ') for c in chip_cols]}")

    chip_col = f"gemm size of {args.chip}"
    if chip_col not in chip_cols:
        print(f"ERROR: chip '{args.chip}' not found. Available: {chip_cols}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Filtering by: {chip_col}")
    print(f"Snippet count: {args.snippet_count}\n")

    for category in CATEGORIES:
        matched = filter_by_category(rows, chip_col, category)
        out_rows = [to_output_row(r) for r in matched]

        full_path = args.output_dir / f"{category}.csv"
        snippet_path = args.output_dir / f"{category}Snippet.csv"

        write_csv(full_path, out_rows)
        write_csv(snippet_path, out_rows[: args.snippet_count])

        print(f"{category}: {len(out_rows)} total → {full_path.name}, "
              f"{min(len(out_rows), args.snippet_count)} snippet → {snippet_path.name}")

    print(f"\nDone. Files written to {args.output_dir}/")


if __name__ == "__main__":
    main()
