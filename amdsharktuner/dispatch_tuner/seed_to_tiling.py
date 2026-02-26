#!/usr/bin/env python3
import argparse
import math
from functools import reduce
from math import gcd
from typing import List, Tuple


def parse_int_list(value: str) -> List[int]:
    if value.strip() == "":
        return []
    return [int(x) for x in value.split(",")]


def product(values: List[int]) -> int:
    return reduce(lambda a, b: a * b, values, 1)


def divide_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def distribute_tiles_using_gcd(total_tiles: int, tiles_to_distribute: int) -> Tuple[int, int, int]:
    dist = gcd(tiles_to_distribute, total_tiles)
    total_tiles //= dist
    tiles_to_distribute //= dist
    return dist, total_tiles, tiles_to_distribute


def distribute_sqrt_for_dim(
    is_m_dim: bool,
    subgroup_sqrt: int,
    tile_sqrt: int,
    m_total: int,
    n_total: int,
    m_subgroup: int,
    n_subgroup: int,
    m_tile: int,
    n_tile: int,
    remaining_subgroups: int,
    remaining_tiles: int,
) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
    if is_m_dim:
        m_subgroup = subgroup_sqrt
        m_tile = tile_sqrt
        m_total //= (subgroup_sqrt * tile_sqrt)
    else:
        n_subgroup = subgroup_sqrt
        n_tile = tile_sqrt
        n_total //= (subgroup_sqrt * tile_sqrt)

    remaining_subgroups //= subgroup_sqrt
    remaining_tiles //= tile_sqrt
    return (
        m_total,
        n_total,
        m_subgroup,
        n_subgroup,
        m_tile,
        n_tile,
        remaining_subgroups,
        remaining_tiles,
    )


def distribute_gcd_for_dim(
    is_m_dim: bool,
    m_total: int,
    n_total: int,
    m_subgroup: int,
    n_subgroup: int,
    m_tile: int,
    n_tile: int,
    remaining_subgroups: int,
    remaining_tiles: int,
) -> Tuple[int, int, int, int, int, int, int, int]:
    if is_m_dim:
        dist, m_total, remaining_subgroups = distribute_tiles_using_gcd(
            m_total, remaining_subgroups
        )
        m_subgroup = dist
        dist, m_total, remaining_tiles = distribute_tiles_using_gcd(
            m_total, remaining_tiles
        )
        m_tile = dist
    else:
        dist, n_total, remaining_subgroups = distribute_tiles_using_gcd(
            n_total, remaining_subgroups
        )
        n_subgroup = dist
        dist, n_total, remaining_tiles = distribute_tiles_using_gcd(
            n_total, remaining_tiles
        )
        n_tile = dist

    return (
        m_total,
        n_total,
        m_subgroup,
        n_subgroup,
        m_tile,
        n_tile,
        remaining_subgroups,
        remaining_tiles,
    )


def get_best_k_tile_sizes(
    k_sizes: List[int],
    intrinsic_k_sizes: List[int],
    best_k_tile_count_per_subgroup: int,
    best_k_element_count_per_subgroup: int,
) -> List[int]:
    k_total_tile_counts = list(k_sizes)
    for i, intrinsic_k_size in enumerate(reversed(intrinsic_k_sizes)):
        idx = len(k_total_tile_counts) - 1 - i
        k_total_tile_counts[idx] = divide_ceil(k_total_tile_counts[idx], intrinsic_k_size)

    if len(intrinsic_k_sizes) > 1:
        best_k_tile_count_per_subgroup = divide_ceil(
            best_k_tile_count_per_subgroup, intrinsic_k_sizes[1]
        )
        best_k_element_count_per_subgroup = divide_ceil(
            best_k_element_count_per_subgroup, intrinsic_k_sizes[1]
        )

    if best_k_element_count_per_subgroup:
        best_k_tile_count_per_subgroup = divide_ceil(
            best_k_element_count_per_subgroup, intrinsic_k_sizes[0]
        )

    k_tile_sizes = [0] * len(k_sizes)
    k_dim = len(k_sizes) - 1
    while k_dim >= 0:
        dist = gcd(k_total_tile_counts[k_dim], best_k_tile_count_per_subgroup)
        k_tile_sizes[k_dim] = dist
        best_k_tile_count_per_subgroup //= dist
        k_dim -= 1
    return k_tile_sizes


def compute_schedule(
    m_sizes: List[int],
    n_sizes: List[int],
    k_sizes: List[int],
    intrinsic_m_sizes: List[int],
    intrinsic_n_sizes: List[int],
    intrinsic_k_sizes: List[int],
    best_subgroup_count_per_workgroup: int,
    best_mn_tile_count_per_subgroup: int,
    best_k_tile_count_per_subgroup: int,
    best_k_element_count_per_subgroup: int,
) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
    m_total_tile_counts = list(m_sizes)
    n_total_tile_counts = list(n_sizes)
    m_total_tile_counts[-1] = divide_ceil(m_sizes[-1], intrinsic_m_sizes[-1])
    n_total_tile_counts[-1] = divide_ceil(n_sizes[-1], intrinsic_n_sizes[-1])

    m_total_to_distribute = product(m_total_tile_counts)
    n_total_to_distribute = product(n_total_tile_counts)

    remaining_subgroups = best_subgroup_count_per_workgroup
    remaining_tiles = best_mn_tile_count_per_subgroup

    m_subgroup = 1
    n_subgroup = 1
    m_tile = 1
    n_tile = 1

    log2_subgroups = int(math.log2(remaining_subgroups))
    subgroup_sqrt = 1 << divide_ceil(log2_subgroups, 2)
    log2_tiles = int(math.log2(remaining_tiles))
    tile_sqrt = 1 << (log2_tiles // 2)
    split_factor = subgroup_sqrt * tile_sqrt

    can_m_distribute_evenly = (
        m_total_to_distribute > split_factor
        and m_total_to_distribute % split_factor == 0
    )
    can_n_distribute_evenly = (
        n_total_to_distribute > split_factor
        and n_total_to_distribute % split_factor == 0
    )

    if can_m_distribute_evenly:
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_sqrt_for_dim(
            True,
            subgroup_sqrt,
            tile_sqrt,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_gcd_for_dim(
            False,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )
    elif can_n_distribute_evenly:
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_sqrt_for_dim(
            False,
            subgroup_sqrt,
            tile_sqrt,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_gcd_for_dim(
            True,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )
    else:
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_gcd_for_dim(
            False,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )
        (
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        ) = distribute_gcd_for_dim(
            True,
            m_total_to_distribute,
            n_total_to_distribute,
            m_subgroup,
            n_subgroup,
            m_tile,
            n_tile,
            remaining_subgroups,
            remaining_tiles,
        )

    m_subgroup_counts = [0] * len(m_sizes)
    n_subgroup_counts = [0] * len(n_sizes)
    m_tile_sizes = [0] * len(m_sizes)
    n_tile_sizes = [0] * len(n_sizes)

    for i in range(len(m_sizes) - 1, -1, -1):
        dist, m_total_tile_counts[i], m_subgroup = distribute_tiles_using_gcd(
            m_total_tile_counts[i], m_subgroup
        )
        m_subgroup_counts[i] = dist
        dist, m_total_tile_counts[i], m_tile = distribute_tiles_using_gcd(
            m_total_tile_counts[i], m_tile
        )
        m_tile_sizes[i] = dist

    for i in range(len(n_sizes) - 1, -1, -1):
        dist, n_total_tile_counts[i], n_subgroup = distribute_tiles_using_gcd(
            n_total_tile_counts[i], n_subgroup
        )
        n_subgroup_counts[i] = dist
        dist, n_total_tile_counts[i], n_tile = distribute_tiles_using_gcd(
            n_total_tile_counts[i], n_tile
        )
        n_tile_sizes[i] = dist

    k_tile_sizes = get_best_k_tile_sizes(
        k_sizes,
        intrinsic_k_sizes,
        best_k_tile_count_per_subgroup,
        best_k_element_count_per_subgroup,
    )

    return m_subgroup_counts, n_subgroup_counts, m_tile_sizes, n_tile_sizes, k_tile_sizes


def compute_lowering_config(
    rank: int,
    m_dims: List[int],
    n_dims: List[int],
    k_dims: List[int],
    batch_dims: List[int],
    intrinsic_m_sizes: List[int],
    intrinsic_n_sizes: List[int],
    m_subgroup_counts: List[int],
    n_subgroup_counts: List[int],
    m_tile_sizes: List[int],
    n_tile_sizes: List[int],
    k_tile_sizes: List[int],
) -> Tuple[List[int], List[int], List[int]]:
    workgroup = [0] * rank
    subgroup = [0] * rank
    reduction = [0] * rank

    for b in batch_dims:
        workgroup[b] = 1

    for i, dim in enumerate(m_dims):
        workgroup[dim] = m_subgroup_counts[i] * m_tile_sizes[i]
        if i == len(m_dims) - 1:
            workgroup[dim] *= product(intrinsic_m_sizes)
        subgroup[dim] = m_tile_sizes[i]

    for i, dim in enumerate(n_dims):
        workgroup[dim] = n_subgroup_counts[i] * n_tile_sizes[i]
        if i == len(n_dims) - 1:
            workgroup[dim] *= product(intrinsic_n_sizes)
        subgroup[dim] = n_tile_sizes[i]

    for i, dim in enumerate(k_dims):
        reduction[dim] = k_tile_sizes[i]

    return workgroup, subgroup, reduction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GPUMMAHeuristicSeeds to lowering_config arrays."
    )
    parser.add_argument("--m-sizes", required=True, type=parse_int_list)
    parser.add_argument("--n-sizes", required=True, type=parse_int_list)
    parser.add_argument("--k-sizes", required=True, type=parse_int_list)
    parser.add_argument("--intrinsic-m", required=True, type=parse_int_list)
    parser.add_argument("--intrinsic-n", required=True, type=parse_int_list)
    parser.add_argument("--intrinsic-k", required=True, type=parse_int_list)
    parser.add_argument("--m-dims", required=True, type=parse_int_list)
    parser.add_argument("--n-dims", required=True, type=parse_int_list)
    parser.add_argument("--k-dims", required=True, type=parse_int_list)
    parser.add_argument("--batch-dims", default="", type=parse_int_list)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--best-subgroup-count-per-workgroup", required=True, type=int)
    parser.add_argument("--best-mn-tile-count-per-subgroup", required=True, type=int)
    parser.add_argument("--best-k-tile-count-per-subgroup", required=True, type=int)
    parser.add_argument("--best-k-element-count-per-subgroup", required=True, type=int)
    args = parser.parse_args()

    (
        m_subgroup_counts,
        n_subgroup_counts,
        m_tile_sizes,
        n_tile_sizes,
        k_tile_sizes,
    ) = compute_schedule(
        args.m_sizes,
        args.n_sizes,
        args.k_sizes,
        args.intrinsic_m,
        args.intrinsic_n,
        args.intrinsic_k,
        args.best_subgroup_count_per_workgroup,
        args.best_mn_tile_count_per_subgroup,
        args.best_k_tile_count_per_subgroup,
        args.best_k_element_count_per_subgroup,
    )

    workgroup, subgroup, reduction = compute_lowering_config(
        args.rank,
        args.m_dims,
        args.n_dims,
        args.k_dims,
        args.batch_dims,
        args.intrinsic_m,
        args.intrinsic_n,
        m_subgroup_counts,
        n_subgroup_counts,
        m_tile_sizes,
        n_tile_sizes,
        k_tile_sizes,
    )

    print("workgroup:", workgroup)
    print("subgroup :", subgroup)
    print("reduction:", reduction)
    print("m_subgroup_counts:", m_subgroup_counts)
    print("n_subgroup_counts:", n_subgroup_counts)
    print("m_tile_sizes:", m_tile_sizes)
    print("n_tile_sizes:", n_tile_sizes)
    print("k_tile_sizes:", k_tile_sizes)


if __name__ == "__main__":
    main()
