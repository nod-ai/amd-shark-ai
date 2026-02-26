#!/usr/bin/env python3
import argparse
from functools import reduce


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
