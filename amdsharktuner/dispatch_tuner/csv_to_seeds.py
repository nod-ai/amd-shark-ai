#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gemm_intensity import compute_intensity


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return val


def _find_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return match.group(1)


def _parse_affine_map_list(text: str) -> List[str]:
    rhs = _find_first(r"->\s*\(([^)]*)\)", text)
    return [s.strip() for s in rhs.split(",") if s.strip()]


def _parse_symbol_list(text: str) -> List[str]:
    lhs = _find_first(r"affine_map<\(([^)]*)\)\s*->", text)
    return [s.strip() for s in lhs.split(",") if s.strip()]


def _parse_tensor_shape(text: str) -> List[int]:
    shape_text = _find_first(r"tensor<([^>]+)>", text)
    parts = shape_text.split("x")
    dims = parts[:-1]
    return [int(d) for d in dims]


def _map_dim_roles(
    symbols: List[str],
    lhs_map: List[str],
    rhs_map: List[str],
    out_map: List[str],
) -> Dict[str, List[int]]:
    lhs_set = set(lhs_map)
    rhs_set = set(rhs_map)
    out_set = set(out_map)

    k_syms = sorted((lhs_set & rhs_set) - out_set, key=symbols.index)
    batch_syms = sorted(lhs_set & rhs_set & out_set, key=symbols.index)
    m_syms = sorted((out_set & lhs_set) - rhs_set, key=symbols.index)
    n_syms = sorted((out_set & rhs_set) - lhs_set, key=symbols.index)

    return {
        "batch": [symbols.index(s) for s in batch_syms],
        "m": [symbols.index(s) for s in m_syms],
        "n": [symbols.index(s) for s in n_syms],
        "k": [symbols.index(s) for s in k_syms],
    }


def _infer_dim_sizes(
    symbols: List[str],
    lhs_map: List[str],
    rhs_map: List[str],
    out_map: List[str],
    lhs_shape: List[int],
    rhs_shape: List[int],
    out_shape: List[int],
) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for sym, dim in zip(out_map, out_shape):
        sizes[sym] = dim
    for sym, dim in zip(lhs_map, lhs_shape):
        sizes.setdefault(sym, dim)
    for sym, dim in zip(rhs_map, rhs_shape):
        sizes.setdefault(sym, dim)
    return sizes


def parse_mlir_contraction_sizes(mlir_path: Path) -> Tuple[List[int], List[int], List[int]]:
    text = mlir_path.read_text(encoding="utf-8")

    op_text = ""
    for line in text.splitlines():
        if "linalg.generic" in line and "indexing_maps" in line:
            op_text = line.strip()
            break
    if not op_text:
        raise ValueError("No linalg.generic with indexing_maps found")

    maps_text = _find_first(r"indexing_maps\s*=\s*\[([^\]]+)\]", op_text)
    map_entries = [s.strip() for s in maps_text.split(">,") if s.strip()]
    if len(map_entries) < 3:
        raise ValueError("Expected at least 3 indexing maps")
    lhs_map = _parse_affine_map_list(
        map_entries[0] + (">" if not map_entries[0].endswith(">") else "")
    )
    rhs_map = _parse_affine_map_list(
        map_entries[1] + (">" if not map_entries[1].endswith(">") else "")
    )
    out_map = _parse_affine_map_list(
        map_entries[2] + (">" if not map_entries[2].endswith(">") else "")
    )

    symbols = _parse_symbol_list(
        map_entries[0] + (">" if not map_entries[0].endswith(">") else "")
    )

    ins_shapes_text = _find_first(r"ins\([^)]*:\s*([^\)]*)\)\s*outs", op_text)
    ins_shapes = [s.strip() for s in ins_shapes_text.split(",") if "tensor<" in s]
    if len(ins_shapes) < 2:
        raise ValueError("Expected at least two input tensors")
    lhs_shape = _parse_tensor_shape(ins_shapes[0])
    rhs_shape = _parse_tensor_shape(ins_shapes[1])

    out_shape_text = _find_first(r"outs\([^)]*:\s*([^\)]*)\)", op_text)
    out_shape = _parse_tensor_shape(out_shape_text)

    dim_roles = _map_dim_roles(symbols, lhs_map, rhs_map, out_map)
    sizes_by_symbol = _infer_dim_sizes(
        symbols, lhs_map, rhs_map, out_map, lhs_shape, rhs_shape, out_shape
    )

    m_sizes = [sizes_by_symbol[symbols[d]] for d in dim_roles["m"]]
    n_sizes = [sizes_by_symbol[symbols[d]] for d in dim_roles["n"]]
    k_sizes = [sizes_by_symbol[symbols[d]] for d in dim_roles["k"]]

    return m_sizes, n_sizes, k_sizes


def derive_seeds_from_knobs(
    tile_k: int,
    subgroup_m_cnt: int,
    subgroup_n_cnt: int,
    subgroup_m: int,
    subgroup_n: int,
    intrinsic_k: int,
) -> Dict[str, int]:
    best_subgroup_count = subgroup_m_cnt * subgroup_n_cnt
    best_mn_tile_count = subgroup_m * subgroup_n
    best_k_tile_count = tile_k
    best_k_element_count = tile_k * intrinsic_k
    return {
        "best_subgroup_count_per_workgroup": best_subgroup_count,
        "best_mn_tile_count_per_subgroup": best_mn_tile_count,
        "best_k_tile_count_per_subgroup": best_k_tile_count,
        "best_k_element_count_per_subgroup": best_k_element_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan tuning CSVs for benchmark_speedup<1 and benchmark_rank_order<10, "
            "then derive heuristic seeds from tiling knobs and MLIR sizes."
        )
    )
    parser.add_argument(
        "--csv-dir",
        default="tuning_database_tf",
        help="Directory containing tuning CSVs (relative to dispatch_tuner).",
    )
    parser.add_argument(
        "--mlir-dir",
        default="conv_dump",
        help="Directory containing corresponding MLIR files (relative to dispatch_tuner).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output CSV path. If not set, prints to stdout.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    csv_dir = (base_dir / args.csv_dir).resolve()
    mlir_dir = (base_dir / args.mlir_dir).resolve()

    rows_out: List[Dict[str, str]] = []

    for csv_path in sorted(csv_dir.glob("*.csv")):
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                speedup = _parse_float(row.get("benchmark_speedup", ""))
                rank = _parse_int(row.get("benchmark_rank_order", ""))
                if speedup is None or rank is None:
                    continue
                if not (speedup < 1.0 and rank < 10):
                    continue

                mlir_path = mlir_dir / csv_path.with_suffix(".mlir").name
                if not mlir_path.exists():
                    continue

                try:
                    m_sizes, n_sizes, k_sizes = parse_mlir_contraction_sizes(mlir_path)
                except ValueError:
                    continue

                intensity = compute_intensity(m_sizes, n_sizes, k_sizes, scaled=False)

                tile_k = _parse_int(row.get("knob_tile_k", "")) or 0
                subgroup_m_cnt = _parse_int(row.get("knob_subgroup_m_cnt", "")) or 0
                subgroup_n_cnt = _parse_int(row.get("knob_subgroup_n_cnt", "")) or 0
                subgroup_m = _parse_int(row.get("knob_subgroup_m", "")) or 0
                subgroup_n = _parse_int(row.get("knob_subgroup_n", "")) or 0
                intrinsic_mn = _parse_int(row.get("knob_intrinsic_mn", "")) or 0
                intrinsic_k = _parse_int(row.get("knob_intrinsic_k", "")) or 0

                seeds = derive_seeds_from_knobs(
                    tile_k=tile_k,
                    subgroup_m_cnt=subgroup_m_cnt,
                    subgroup_n_cnt=subgroup_n_cnt,
                    subgroup_m=subgroup_m,
                    subgroup_n=subgroup_n,
                    intrinsic_k=intrinsic_k,
                )

                rows_out.append(
                    {
                        "csv_file": csv_path.name,
                        "mlir_file": mlir_path.name,
                        "gen_id": row.get("gen_id", ""),
                        "candidate_id": row.get("candidate_id", ""),
                        "benchmark_speedup": f"{speedup}",
                        "benchmark_rank_order": f"{rank}",
                        "M_sizes": ",".join(str(x) for x in m_sizes),
                        "N_sizes": ",".join(str(x) for x in n_sizes),
                        "K_sizes": ",".join(str(x) for x in k_sizes),
                        "compute_intensity": f"{intensity}",
                        "knob_tile_m": row.get("knob_tile_m", ""),
                        "knob_tile_n": row.get("knob_tile_n", ""),
                        "knob_tile_k": row.get("knob_tile_k", ""),
                        "knob_subgroup_m_cnt": row.get("knob_subgroup_m_cnt", ""),
                        "knob_subgroup_n_cnt": row.get("knob_subgroup_n_cnt", ""),
                        "knob_subgroup_m": row.get("knob_subgroup_m", ""),
                        "knob_subgroup_n": row.get("knob_subgroup_n", ""),
                        "knob_intrinsic_mn": f"{intrinsic_mn}",
                        "knob_intrinsic_k": f"{intrinsic_k}",
                        **{k: str(v) for k, v in seeds.items()},
                    }
                )

    if not rows_out:
        return

    headers = list(rows_out[0].keys())

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows_out)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows_out)


if __name__ == "__main__":
    main()
