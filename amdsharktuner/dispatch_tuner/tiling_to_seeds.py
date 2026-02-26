#!/usr/bin/env python3
"""Extract lowering_config from an MLIR file and find all valid seed combinations.

Supports two MLIR file formats:
  - benchmark files (contain hal.executable / linalg.generic)
  - tuned spec files (contain iree_codegen.tuning_spec)

Usage:
  python tiling_to_seeds.py <path_to_mlir_file>
  python tiling_to_seeds.py <path_to_mlir_file> --json
"""

import argparse
import json
import re
import sys
from math import gcd
from typing import Dict, List, Optional, Tuple

from seed_to_tiling import (
    compute_lowering_config,
    compute_schedule,
    divide_ceil,
    product,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_int_list_from_brackets(s: str) -> List[int]:
    """Parse '[1, 2, 3]' -> [1, 2, 3]."""
    inner = s.strip().strip("[]")
    if not inner.strip():
        return []
    return [int(x) for x in inner.split(",")]


def _parse_mma_intrinsic(mma_str: str) -> Tuple[List[int], List[int], List[int]]:
    """Parse 'MFMA_F32_16x16x16_BF16' -> intrinsic (m, n, k) sizes."""
    m = re.search(r"MFMA_\w+?_(\d+)x(\d+)x(\d+)", mma_str)
    if not m:
        raise ValueError(f"Cannot parse MMA intrinsic from: {mma_str}")
    return [int(m.group(1))], [int(m.group(2))], [int(m.group(3))]


def _extract_lowering_config(text: str) -> Dict:
    """Extract workgroup, subgroup, reduction, mma_kind from lowering_config."""
    lc = re.search(r"lowering_config\s*=\s*#iree_gpu\.lowering_config<\{(.+?)\}>", text)
    if not lc:
        raise ValueError("No lowering_config found in file")
    body = lc.group(1)

    def grab(key):
        m = re.search(rf"{key}\s*=\s*\[([^\]]*)\]", body)
        if not m:
            raise ValueError(f"Missing '{key}' in lowering_config")
        return _parse_int_list_from_brackets(f"[{m.group(1)}]")

    mma_m = re.search(r"mma_kind\s*=\s*#iree_gpu\.mma_layout<(\w+)>", body)
    if not mma_m:
        raise ValueError("Missing mma_kind in lowering_config")

    return {
        "workgroup": grab("workgroup"),
        "subgroup": grab("subgroup"),
        "reduction": grab("reduction"),
        "mma_kind": mma_m.group(1),
    }


# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------

def detect_file_type(text: str) -> str:
    if "iree_codegen.tuning_spec" in text:
        return "tuned_spec"
    if "linalg.generic" in text or "hal.executable" in text:
        return "benchmark"
    raise ValueError("Cannot detect MLIR file type")


# ---------------------------------------------------------------------------
# Problem parsing: tuned spec
# ---------------------------------------------------------------------------

def _parse_tuned_spec(text: str) -> Dict:
    """Parse M, N, K, batch dims and their sizes from a tuned spec file."""
    dims_pattern = re.compile(
        r"match\.dims_equal\s+%(\w+),\s*\[([^\]]*)\]"
    )
    dim_sizes: Dict[str, List[int]] = {}
    for m in dims_pattern.finditer(text):
        name = m.group(1)
        vals = m.group(2).strip()
        dim_sizes[name] = [int(x) for x in vals.split(",")] if vals else []

    maps_match = re.search(
        r"match\.contraction.*?indexing_maps\s*=\s*\[(.*?)\]\s*:",
        text, re.DOTALL,
    )
    if not maps_match:
        raise ValueError("Cannot find indexing_maps in tuned spec")
    maps_str = maps_match.group(1)
    rank = _rank_from_maps(maps_str)

    m_sizes = dim_sizes.get("m_dims", [])
    n_sizes = dim_sizes.get("n_dims", [])
    k_sizes = dim_sizes.get("k_dims", [])
    batch_sizes = dim_sizes.get("batch_dims", [])

    batch_dims, m_dims, n_dims, k_dims = _classify_dims_from_maps(maps_str, rank)

    return {
        "m_sizes": m_sizes, "n_sizes": n_sizes, "k_sizes": k_sizes,
        "batch_sizes": batch_sizes,
        "m_dims": m_dims, "n_dims": n_dims, "k_dims": k_dims,
        "batch_dims": batch_dims, "rank": rank,
    }


# ---------------------------------------------------------------------------
# Problem parsing: benchmark
# ---------------------------------------------------------------------------

def _parse_benchmark(text: str) -> Dict:
    """Parse M, N, K, batch dims and their sizes from a benchmark MLIR file."""
    generic = re.search(
        r"linalg\.generic\s*\{(.*?)\}.*?ins\((.*?)\)\s*outs\((.*?)\)",
        text, re.DOTALL,
    )
    if not generic:
        raise ValueError("Cannot find linalg.generic in benchmark file")
    attrs_str = generic.group(1)
    ins_str = generic.group(2)
    outs_str = generic.group(3)

    maps_match = re.search(
        r"indexing_maps\s*=\s*\[(.*?)\]",
        attrs_str, re.DOTALL,
    )
    if not maps_match:
        raise ValueError("Cannot find indexing_maps in linalg.generic")
    maps_str = maps_match.group(1)
    rank = _rank_from_maps(maps_str)
    batch_dims, m_dims, n_dims, k_dims = _classify_dims_from_maps(maps_str, rank)

    all_tensors = ins_str + ", " + outs_str
    shapes = re.findall(r"tensor<([\dx]+)x\w+", all_tensors)
    dim_to_size: Dict[int, int] = {}

    maps = _parse_affine_maps(maps_str)
    for tensor_idx, shape_str in enumerate(shapes):
        if tensor_idx >= len(maps):
            break
        sizes = [int(x) for x in shape_str.split("x")]
        amap = maps[tensor_idx]
        for pos, dim_idx in enumerate(amap):
            if pos < len(sizes):
                dim_to_size[dim_idx] = sizes[pos]

    m_sizes = [dim_to_size[d] for d in m_dims]
    n_sizes = [dim_to_size[d] for d in n_dims]
    k_sizes = [dim_to_size[d] for d in k_dims]
    batch_sizes = [dim_to_size[d] for d in batch_dims]

    return {
        "m_sizes": m_sizes, "n_sizes": n_sizes, "k_sizes": k_sizes,
        "batch_sizes": batch_sizes,
        "m_dims": m_dims, "n_dims": n_dims, "k_dims": k_dims,
        "batch_dims": batch_dims, "rank": rank,
    }


# ---------------------------------------------------------------------------
# Indexing map helpers
# ---------------------------------------------------------------------------

def _rank_from_maps(maps_str: str) -> int:
    m = re.search(r"\(([^)]+)\)\s*->", maps_str)
    if not m:
        raise ValueError("Cannot determine rank from indexing maps")
    return len(m.group(1).split(","))


def _parse_affine_maps(maps_str: str) -> List[List[int]]:
    """Parse affine maps and return list of dim-index lists for each map."""
    results = []
    for m in re.finditer(r"->\s*\(([^)]+)\)", maps_str):
        dims = []
        for d in m.group(1).split(","):
            dm = re.search(r"d(\d+)", d.strip())
            if dm:
                dims.append(int(dm.group(1)))
        results.append(dims)
    return results


def _classify_dims_from_maps(
    maps_str: str, rank: int
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Classify dims into batch, M, N, K from contraction indexing maps."""
    maps = _parse_affine_maps(maps_str)
    if len(maps) < 3:
        raise ValueError(f"Expected 3 indexing maps, got {len(maps)}")

    lhs_dims = set(maps[0])
    rhs_dims = set(maps[1])
    out_dims = set(maps[2])

    batch_dims, m_dims, n_dims, k_dims = [], [], [], []
    for d in range(rank):
        in_lhs = d in lhs_dims
        in_rhs = d in rhs_dims
        in_out = d in out_dims
        if in_lhs and in_rhs and in_out:
            batch_dims.append(d)
        elif in_lhs and in_out and not in_rhs:
            m_dims.append(d)
        elif in_rhs and in_out and not in_lhs:
            n_dims.append(d)
        elif in_lhs and in_rhs and not in_out:
            k_dims.append(d)
        else:
            batch_dims.append(d)

    return batch_dims, m_dims, n_dims, k_dims


# ---------------------------------------------------------------------------
# Seed search
# ---------------------------------------------------------------------------

def find_valid_seeds(
    workgroup: List[int],
    subgroup: List[int],
    reduction: List[int],
    m_sizes: List[int],
    n_sizes: List[int],
    k_sizes: List[int],
    intrinsic_m: List[int],
    intrinsic_n: List[int],
    intrinsic_k: List[int],
    m_dims: List[int],
    n_dims: List[int],
    k_dims: List[int],
    batch_dims: List[int],
    rank: int,
    sg_range: Optional[List[int]] = None,
    mn_range: Optional[List[int]] = None,
    kt_max: int = 32,
    ke_max: int = 512,
) -> List[Dict[str, int]]:
    """Brute-force find all seed 4-tuples producing the given tiling."""
    if sg_range is None:
        sg_range = [1 << i for i in range(6)]
    if mn_range is None:
        mn_range = [1 << i for i in range(6)]

    check_mn_dims = m_dims + n_dims + batch_dims

    valid_mn = set()
    for best_sg in sg_range:
        for best_mn in mn_range:
            try:
                r = compute_schedule(
                    m_sizes, n_sizes, k_sizes,
                    intrinsic_m, intrinsic_n, intrinsic_k,
                    best_sg, best_mn, 1, 0,
                )
                wg, sg, _ = compute_lowering_config(
                    rank, m_dims, n_dims, k_dims, batch_dims,
                    intrinsic_m, intrinsic_n, *r,
                )
                if all(wg[d] == workgroup[d] and sg[d] == subgroup[d]
                       for d in check_mn_dims):
                    valid_mn.add((best_sg, best_mn))
            except Exception:
                pass

    k_total_counts = list(k_sizes)
    for i, ik in enumerate(reversed(intrinsic_k)):
        idx = len(k_total_counts) - 1 - i
        k_total_counts[idx] = divide_ceil(k_total_counts[idx], ik)

    target_k_tiles = [reduction[d] for d in k_dims]

    valid_k = set()
    for best_kt in range(1, kt_max + 1):
        for best_ke in [0] + list(range(1, ke_max + 1)):
            eff = best_kt
            if len(intrinsic_k) > 1:
                eff_kt = divide_ceil(best_kt, intrinsic_k[1])
                eff_ke = divide_ceil(best_ke, intrinsic_k[1]) if best_ke else 0
            else:
                eff_kt = best_kt
                eff_ke = best_ke
            eff = divide_ceil(eff_ke, intrinsic_k[0]) if eff_ke else eff_kt

            computed_k = []
            remaining = eff
            for ki in range(len(k_sizes) - 1, -1, -1):
                g = gcd(k_total_counts[ki], remaining)
                computed_k.insert(0, g)
                remaining //= g

            if computed_k == target_k_tiles:
                valid_k.add((best_kt, best_ke))

    results = []
    for sg_val, mn_val in sorted(valid_mn):
        for kt_val, ke_val in sorted(valid_k):
            try:
                r = compute_schedule(
                    m_sizes, n_sizes, k_sizes,
                    intrinsic_m, intrinsic_n, intrinsic_k,
                    sg_val, mn_val, kt_val, ke_val,
                )
                wg, sg, red = compute_lowering_config(
                    rank, m_dims, n_dims, k_dims, batch_dims,
                    intrinsic_m, intrinsic_n, *r,
                )
                if wg == workgroup and sg == subgroup and red == reduction:
                    results.append({
                        "best_subgroup_count_per_workgroup": sg_val,
                        "best_mn_tile_count_per_subgroup": mn_val,
                        "best_k_tile_count_per_subgroup": kt_val,
                        "best_k_element_count_per_subgroup": ke_val,
                    })
            except Exception:
                pass

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_mlir_file(filepath: str) -> Dict:
    """Read an MLIR file, extract tiling, find all valid seeds.

    Returns a dict with keys:
      file_type, lowering_config, problem, valid_seeds, valid_seed_count,
      unique_sg, unique_mn
    """
    with open(filepath) as f:
        text = f.read()

    file_type = detect_file_type(text)
    config = _extract_lowering_config(text)
    intrinsic_m, intrinsic_n, intrinsic_k = _parse_mma_intrinsic(config["mma_kind"])

    if file_type == "tuned_spec":
        problem = _parse_tuned_spec(text)
    else:
        problem = _parse_benchmark(text)

    seeds = find_valid_seeds(
        config["workgroup"], config["subgroup"], config["reduction"],
        problem["m_sizes"], problem["n_sizes"], problem["k_sizes"],
        intrinsic_m, intrinsic_n, intrinsic_k,
        problem["m_dims"], problem["n_dims"], problem["k_dims"],
        problem["batch_dims"], problem["rank"],
    )

    unique_sg = sorted(set(s["best_subgroup_count_per_workgroup"] for s in seeds))
    unique_mn = sorted(set(s["best_mn_tile_count_per_subgroup"] for s in seeds))

    return {
        "filepath": filepath,
        "file_type": file_type,
        "lowering_config": config,
        "problem": problem,
        "valid_seeds": seeds,
        "valid_seed_count": len(seeds),
        "unique_sg_values": unique_sg,
        "unique_mn_values": unique_mn,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract tiling from MLIR and find all valid seeds."
    )
    parser.add_argument("mlir_file", help="Path to an MLIR file")
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON instead of human-readable text",
    )
    args = parser.parse_args()

    result = analyze_mlir_file(args.mlir_file)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        cfg = result["lowering_config"]
        prob = result["problem"]
        print(f"File: {result['filepath']}")
        print(f"Type: {result['file_type']}")
        print(f"MMA:  {cfg['mma_kind']}")
        print(f"M={prob['m_sizes']} N={prob['n_sizes']} K={prob['k_sizes']} "
              f"batch={prob['batch_sizes']}")
        print(f"workgroup={cfg['workgroup']}  subgroup={cfg['subgroup']}  "
              f"reduction={cfg['reduction']}")
        print(f"\nValid seeds: {result['valid_seed_count']} combinations")
        print(f"  Unique best_subgroup values: {result['unique_sg_values']}")
        print(f"  Unique best_mn_tile values:  {result['unique_mn_values']}")
        if result["valid_seed_count"] <= 20:
            for s in result["valid_seeds"]:
                print(f"    sg={s['best_subgroup_count_per_workgroup']}, "
                      f"mn={s['best_mn_tile_count_per_subgroup']}, "
                      f"kt={s['best_k_tile_count_per_subgroup']}, "
                      f"ke={s['best_k_element_count_per_subgroup']}")
        else:
            for s in result["valid_seeds"][:5]:
                print(f"    sg={s['best_subgroup_count_per_workgroup']}, "
                      f"mn={s['best_mn_tile_count_per_subgroup']}, "
                      f"kt={s['best_k_tile_count_per_subgroup']}, "
                      f"ke={s['best_k_element_count_per_subgroup']}")
            print(f"    ... and {result['valid_seed_count'] - 5} more")


if __name__ == "__main__":
    main()
