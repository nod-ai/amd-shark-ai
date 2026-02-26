#!/usr/bin/env python3
"""Scan tuning directories and match against input CSV filenames
to determine which dispatches were improved by tuning.

Usage:
    python check_tuning_results.py --input path/to/input.csv [--output path/to/output.json]
    python check_tuning_results.py --batch-dir path/to/SMLGemm/

When --output is omitted, the output is written next to --input as
    {input_stem}_tuning_check_results.json

When --batch-dir is given, all non-Snippet CSVs in that directory are
processed in parallel using half the available CPU cores.
"""

import argparse
import csv
import json
import multiprocessing
import os
import re
from pathlib import Path

TUNER_ROOT = Path(__file__).resolve().parent.parent
CONV_DUMP_DIR = Path(__file__).resolve().parent / "conv_dump"

BETTER_THRESHOLD = 90.0

CANDIDATE_RE = re.compile(
    r"Candidate\s+(\d+)\s+time:\s+([\d.]+)\s+us\s+\(([\d.]+)%\s+of\s+baseline\)"
)


def load_tuning_subdirs(root: Path) -> list[Path]:
    """Return tuning_2026_02* subdirs that contain at least one .csv file."""
    subdirs = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("tuning_2026_02"):
            continue
        if any(f.suffix == ".csv" for f in entry.iterdir() if f.is_file()):
            subdirs.append(entry)
    return subdirs


def build_csv_index(subdirs: list[Path]) -> dict[str, list[tuple[Path, str]]]:
    """Map each tuning subdir to its csv filenames (stem only, no extension).

    Returns {csv_stem: [(subdir, csv_filename), ...]}.
    """
    index: dict[str, list[tuple[Path, str]]] = {}
    for subdir in subdirs:
        for f in subdir.iterdir():
            if f.is_file() and f.suffix == ".csv":
                csv_stem = f.stem
                index.setdefault(csv_stem, []).append((subdir, f.name))
    return index


def find_tuning_dir(
    filename_stem: str,
    csv_index: dict[str, list[tuple[Path, str]]],
) -> Path | None:
    """Find the first tuning subdir whose csv filename contains *filename_stem*."""
    matches: list[tuple[Path, str]] = []
    for csv_stem, entries in csv_index.items():
        if filename_stem in csv_stem:
            matches.extend(entries)
    if not matches:
        return None
    if len(matches) > 1:
        dirs = sorted(set(str(e[0]) for e in matches))
        print(
            f"  WARNING: {len(matches)} CSV match(es) across dirs for "
            f"'{filename_stem}': {dirs}. Picking first."
        )
    return min(matches, key=lambda e: e[0].name)[0]


def parse_summary_log(
    tuning_dir: Path,
) -> tuple[list[tuple[int, float]], float | None, int | None]:
    """Parse summary.log and return (qualifying_candidates, best_pct, best_id).

    qualifying_candidates contains all (candidate_id, pct) pairs with pct < BETTER_THRESHOLD,
    sorted by pct ascending.
    """
    log_path = tuning_dir / "summary.log"
    if not log_path.exists():
        print(f"  WARNING: summary.log missing in {tuning_dir}")
        return [], None, None

    all_candidates: list[tuple[int, float]] = []
    with open(log_path) as f:
        for line in f:
            m = CANDIDATE_RE.search(line)
            if m:
                cand_id = int(m.group(1))
                pct = float(m.group(3))
                all_candidates.append((cand_id, pct))

    if not all_candidates:
        print(f"  WARNING: No candidate results found in {log_path}")
        return [], None, None

    best_id, best_pct = min(all_candidates, key=lambda x: x[1])
    qualifying = sorted(
        [(cid, p) for cid, p in all_candidates if p < BETTER_THRESHOLD],
        key=lambda x: x[1],
    )

    if not qualifying:
        print(
            f"  No Better: best candidate {best_id} at {best_pct}% "
            f"(>= {BETTER_THRESHOLD}% threshold)"
        )

    return qualifying, best_pct, best_id


def find_benchmark_file(filename_stem: str) -> str | None:
    """Find a file in conv_dump whose name contains the filename stem."""
    if not CONV_DUMP_DIR.is_dir():
        return None
    for f in CONV_DUMP_DIR.iterdir():
        if f.is_file() and filename_stem in f.name:
            return str(f)
    return None


def find_spec_path(tuning_dir: Path, candidate_id: int) -> str | None:
    spec = tuning_dir / "candidates" / "specs" / f"{candidate_id}_spec.mlir"
    if spec.exists():
        return str(spec)
    print(f"  WARNING: spec file not found: {spec}")
    return None


def derive_output_path(input_csv: Path, output_dir: Path | None = None) -> Path:
    """Derive output JSON path from input CSV path."""
    name = f"{input_csv.stem}_tuning_check_results.json"
    if output_dir:
        return output_dir / name
    return input_csv.parent / name


def process_single(
    input_csv: Path,
    output_json: Path,
    csv_index: dict[str, list[tuple[Path, str]]],
) -> None:
    """Process one input CSV and write results to output_json."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_csv}")
    print(f"Output:     {output_json}")
    print(f"{'='*60}")

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)
        snippet_rows = list(reader)

    filenames = [row["filename"] for row in snippet_rows]
    proxy_shapes_map = {row["filename"]: row.get("proxy_shapes", "") for row in snippet_rows}
    print(f"Read {len(filenames)} filenames\n")

    results: list[dict] = []
    for fname in filenames:
        stem = Path(fname).stem
        print(f"Processing: {fname}")

        bench_path = find_benchmark_file(stem)
        proxy_shapes = proxy_shapes_map.get(fname, "")

        tuning_dir = find_tuning_dir(stem, csv_index)
        if tuning_dir is None:
            print(f"  No matching tuning dir found")
            results.append({
                "filename": fname,
                "benchmark_file_path": bench_path,
                "proxy_shapes": proxy_shapes,
                "tuning_dir_path": None,
                "Better": None,
                "best_pct_of_baseline": None,
                "tuned_spec_paths": None,
            })
            continue

        print(f"  Tuning dir: {tuning_dir}")
        qualifying, best_pct, best_id = parse_summary_log(tuning_dir)

        is_better = len(qualifying) > 0
        spec_paths: list[dict] = []
        if is_better:
            for cid, pct in qualifying:
                sp = find_spec_path(tuning_dir, cid)
                if sp:
                    spec_paths.append({"candidate_id": cid, "pct_of_baseline": pct, "spec_path": sp})
            print(f"  Better: {len(spec_paths)} candidate(s) under {BETTER_THRESHOLD}%")

        results.append({
            "filename": fname,
            "benchmark_file_path": bench_path,
            "proxy_shapes": proxy_shapes,
            "tuning_dir_path": str(tuning_dir),
            "Better": is_better,
            "best_pct_of_baseline": best_pct,
            "tuned_spec_paths": spec_paths if spec_paths else None,
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    better_count = sum(1 for r in results if r["Better"] is True)
    no_better = sum(1 for r in results if r["Better"] is False)
    no_dir = sum(1 for r in results if r["tuning_dir_path"] is None)
    print(f"\nResults written to: {output_json}")
    print(f"Summary: {better_count} better, {no_better} not better, {no_dir} no tuning dir")


def _worker(args: tuple[Path, Path, dict]) -> str:
    input_csv, output_json, csv_index = args
    process_single(input_csv, output_json, csv_index)
    return f"Done: {input_csv.name} -> {output_json.name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single input CSV file")
    group.add_argument("--batch-dir", type=Path, help="Directory of CSVs to process (excludes *Snippet*)")
    parser.add_argument("--output", type=Path, help="Output JSON path (single mode only)")
    args = parser.parse_args()

    print(f"Loading tuning subdirs from: {TUNER_ROOT}")
    subdirs = load_tuning_subdirs(TUNER_ROOT)
    print(f"Found {len(subdirs)} qualifying tuning_2026_02* subdirs")

    csv_index = build_csv_index(subdirs)
    print(f"Indexed {len(csv_index)} CSV files across subdirs")

    if args.input:
        output = args.output or derive_output_path(args.input)
        process_single(args.input, output, csv_index)
    else:
        batch_dir = args.batch_dir
        input_csvs = sorted([
            f for f in batch_dir.iterdir()
            if f.is_file() and f.suffix == ".csv" and "Snippet" not in f.name
        ])
        if not input_csvs:
            print(f"No non-Snippet CSVs found in {batch_dir}")
            return

        tasks = []
        for csv_path in input_csvs:
            output_json = derive_output_path(csv_path, output_dir=batch_dir)
            tasks.append((csv_path, output_json, csv_index))

        n_workers = max(1, os.cpu_count() // 2)
        print(f"\nBatch mode: {len(tasks)} CSV(s), {n_workers} workers")
        for csv_path in input_csvs:
            print(f"  - {csv_path.name}")

        with multiprocessing.Pool(n_workers) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                print(f"\n>>> {result}")

        print(f"\nAll batch jobs complete.")


if __name__ == "__main__":
    main()
