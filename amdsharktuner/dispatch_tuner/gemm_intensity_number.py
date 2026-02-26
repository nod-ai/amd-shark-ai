#!/usr/bin/env python3
import argparse
import re
from functools import reduce
from typing import Dict


def parse_int_list(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(x) for x in value.split(",")]


def product(values: list[int]) -> int:
    return reduce(lambda a, b: a * b, values, 1)


def compute_cutoffs(perf_tflops: float | None,
                    mem_bw_tbps: float | None,
                    scaled: bool,
                    use_arch_cutoffs: bool) -> tuple[float, float]:
    if scaled:
        return 100.0, 10000.0
    if not use_arch_cutoffs:
        return 1.0, 1000.0
    if perf_tflops is None or mem_bw_tbps is None:
        return 1.0, 1000.0
    compute_memory_cutoff = perf_tflops / mem_bw_tbps
    small = 0.05 * compute_memory_cutoff
    large = 5.0 * compute_memory_cutoff
    return small, large


def compute_intensity(m_sizes: list[int],
                      n_sizes: list[int],
                      k_sizes: list[int],
                      scaled: bool) -> float:
    m_size = product(m_sizes)
    n_size = product(n_sizes)
    k_size = product(k_sizes)

    flops = 2 * m_size * n_size * k_size
    bytes_elems = m_size * n_size + n_size * k_size + m_size * k_size

    # Only support blocking along the last dimension for now.
    outer_k = k_size // k_sizes[-1]
    scales_bytes = m_size * outer_k + n_size * outer_k
    if scaled:
        bytes_elems += scales_bytes

    # Matches the integer division in ConfigUtils.cpp.
    return flops / bytes_elems


def classify_gemm(intensity: float, small: float, large: float) -> str:
    if intensity <= small:
        return "SmallGemm"
    if intensity >= large:
        return "LargeGemm"
    return "MediumGemm"


def _find_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return match.group(1)


def _parse_affine_map_list(text: str) -> list[str]:
    rhs = _find_first(r"->\s*\(([^)]*)\)", text)
    return [s.strip() for s in rhs.split(",") if s.strip()]


def _parse_symbol_list(text: str) -> list[str]:
    lhs = _find_first(r"affine_map<\(([^)]*)\)\s*->", text)
    return [s.strip() for s in lhs.split(",") if s.strip()]


def _parse_tensor_shape(text: str) -> list[int]:
    shape_text = _find_first(r"tensor<([^>]+)>", text)
    parts = shape_text.split("x")
    dims = parts[:-1]
    return [int(d) for d in dims]


def _map_dim_roles(
    symbols: list[str],
    lhs_map: list[str],
    rhs_map: list[str],
    out_map: list[str],
) -> Dict[str, list[int]]:
    lhs_set = set(lhs_map)
    rhs_set = set(rhs_map)
    out_set = set(out_map)

    k_syms = sorted((lhs_set & rhs_set) - out_set, key=symbols.index)
    m_syms = sorted((out_set & lhs_set) - rhs_set, key=symbols.index)
    n_syms = sorted((out_set & rhs_set) - lhs_set, key=symbols.index)

    return {
        "m": [symbols.index(s) for s in m_syms],
        "n": [symbols.index(s) for s in n_syms],
        "k": [symbols.index(s) for s in k_syms],
    }


def _infer_dim_sizes(
    symbols: list[str],
    lhs_map: list[str],
    rhs_map: list[str],
    out_map: list[str],
    lhs_shape: list[int],
    rhs_shape: list[int],
    out_shape: list[int],
) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for sym, dim in zip(out_map, out_shape):
        sizes[sym] = dim
    for sym, dim in zip(lhs_map, lhs_shape):
        sizes.setdefault(sym, dim)
    for sym, dim in zip(rhs_map, rhs_shape):
        sizes.setdefault(sym, dim)
    return sizes


def extract_mnk_from_mlir(mlir_path: str) -> tuple[list[int], list[int], list[int]]:
    text = ""
    with open(mlir_path, "r", encoding="utf-8") as f:
        text = f.read()

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute arithmetic intensity and GEMM size classification."
    )
    parser.add_argument("--m-sizes", required=True, type=parse_int_list)
    parser.add_argument("--n-sizes", required=True, type=parse_int_list)
    parser.add_argument("--k-sizes", required=True, type=parse_int_list)
    parser.add_argument("--perf-tflops", type=float, default=None,
                        help="Target peak performance in TFlops for compute bitwidth.")
    parser.add_argument("--mem-bw-tbps", type=float, default=None,
                        help="Target memory bandwidth in Tbps.")
    parser.add_argument("--use-arch-cutoffs", action="store_true",
                        help="Use perf/mem to compute cutoffs; default uses fixed cutoffs.")
    parser.add_argument("--scaled", action="store_true",
                        help="Use scaled matmul cutoffs.")
    args = parser.parse_args()

    if not args.k_sizes:
        raise ValueError("k-sizes must be non-empty")

    intensity = compute_intensity(args.m_sizes, args.n_sizes, args.k_sizes, args.scaled)
    small, large = compute_cutoffs(
        args.perf_tflops,
        args.mem_bw_tbps,
        args.scaled,
        args.use_arch_cutoffs,
    )
    gemm_size = classify_gemm(intensity, small, large)

    print(f"compute_intensity: {intensity}")
    print(f"small_gemm_cutoff: {small}")
    print(f"large_gemm_cutoff: {large}")
    print(f"gemm_size: {gemm_size}")


if __name__ == "__main__":
    main()
