#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip() != ""]


def find_first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"Pattern not found: {pattern}")
    return match.group(1)


def parse_affine_map_list(text: str) -> List[str]:
    # Extract the "-> (...)" rhs and return a list of symbols in order.
    rhs = find_first(r"->\s*\(([^)]*)\)", text)
    return [s.strip() for s in rhs.split(",") if s.strip()]


def parse_symbol_list(text: str) -> List[str]:
    lhs = find_first(r"affine_map<\(([^)]*)\)\s*->", text)
    return [s.strip() for s in lhs.split(",") if s.strip()]


def parse_tensor_shape(text: str) -> List[int]:
    shape_text = find_first(r"tensor<([^>]+)>", text)
    # Strip element type: split by 'x' and drop last if it's type.
    parts = shape_text.split("x")
    # Last part contains element type (e.g. bf16, f32).
    dims = parts[:-1]
    return [int(d) for d in dims]


def product(values: List[int]) -> int:
    result = 1
    for v in values:
        result *= v
    return result


def parse_mma_intrinsic_sizes(text: str) -> Tuple[List[int], List[int], List[int]]:
    # Extract first "MxNxK" triple from mma_layout.
    match = re.search(r"mma_layout<[^>]*?(\d+)x(\d+)x(\d+)[^>]*?>", text)
    if not match:
        raise ValueError("Failed to parse mma_layout intrinsic sizes")
    m, n, k = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return [m], [n], [k]


def map_dim_roles(
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


def infer_dim_sizes(
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


@dataclass
class TilingInfo:
    workgroup: List[int]
    subgroup: List[int]
    reduction: List[int]
    m_subgroup_counts: List[int]
    n_subgroup_counts: List[int]
    m_tile_sizes: List[int]
    n_tile_sizes: List[int]
    k_tile_sizes: List[int]


def extract_tiling_from_mlir(mlir_path: str) -> TilingInfo:
    text = ""
    with open(mlir_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Grab the first linalg.generic line that contains iree_gpu.lowering_config.
    op_text = ""
    for line in text.splitlines():
        if "linalg.generic" in line and "lowering_config" in line:
            op_text = line.strip()
            break
    if not op_text:
        raise ValueError("No linalg.generic with iree_gpu.lowering_config found")

    # Parse maps and shapes.
    maps_text = find_first(r"indexing_maps\s*=\s*\[([^\]]+)\]", op_text)
    map_entries = [s.strip() for s in maps_text.split(">,") if s.strip()]
    if len(map_entries) < 3:
        raise ValueError("Expected at least 3 indexing maps")
    lhs_map = parse_affine_map_list(map_entries[0] + (">" if not map_entries[0].endswith(">") else ""))
    rhs_map = parse_affine_map_list(map_entries[1] + (">" if not map_entries[1].endswith(">") else ""))
    out_map = parse_affine_map_list(map_entries[2] + (">" if not map_entries[2].endswith(">") else ""))

    symbols = parse_symbol_list(map_entries[0] + (">" if not map_entries[0].endswith(">") else ""))

    ins_shapes_text = find_first(r"ins\([^)]*:\s*([^\)]*)\)\s*outs", op_text)
    ins_shapes = [s.strip() for s in ins_shapes_text.split(",") if "tensor<" in s]
    if len(ins_shapes) < 2:
        raise ValueError("Expected at least two input tensors")
    lhs_shape = parse_tensor_shape(ins_shapes[0])
    rhs_shape = parse_tensor_shape(ins_shapes[1])

    out_shape_text = find_first(r"outs\([^)]*:\s*([^\)]*)\)", op_text)
    out_shape = parse_tensor_shape(out_shape_text)

    # Parse lowering_config arrays and intrinsic sizes.
    lowering_text = find_first(r"lowering_config\s*=\s*#iree_gpu\.lowering_config<\{([^}]*)\}>", op_text)
    workgroup = parse_int_list(find_first(r"workgroup\s*=\s*\[([^\]]*)\]", lowering_text))
    subgroup = parse_int_list(find_first(r"subgroup\s*=\s*\[([^\]]*)\]", lowering_text))
    reduction = parse_int_list(find_first(r"reduction\s*=\s*\[([^\]]*)\]", lowering_text))
    intrinsic_m, intrinsic_n, intrinsic_k = parse_mma_intrinsic_sizes(op_text)

    dim_roles = map_dim_roles(symbols, lhs_map, rhs_map, out_map)
    sizes_by_symbol = infer_dim_sizes(
        symbols, lhs_map, rhs_map, out_map, lhs_shape, rhs_shape, out_shape
    )

    m_dims = dim_roles["m"]
    n_dims = dim_roles["n"]
    k_dims = dim_roles["k"]
    batch_dims = dim_roles["batch"]

    # Compute schedule counts from lowering_config and intrinsic sizes.
    m_tile_sizes = [subgroup[d] if d < len(subgroup) else 0 for d in m_dims]
    n_tile_sizes = [subgroup[d] if d < len(subgroup) else 0 for d in n_dims]
    k_tile_sizes = [reduction[d] if d < len(reduction) else 0 for d in k_dims]

    intrinsic_m_total = product(intrinsic_m)
    intrinsic_n_total = product(intrinsic_n)

    m_subgroup_counts = []
    for idx, d in enumerate(m_dims):
        if d < len(workgroup) and m_tile_sizes[idx] > 0 and intrinsic_m_total > 0:
            m_subgroup_counts.append(workgroup[d] // (m_tile_sizes[idx] * intrinsic_m_total))
        else:
            m_subgroup_counts.append(0)

    n_subgroup_counts = []
    for idx, d in enumerate(n_dims):
        if d < len(workgroup) and n_tile_sizes[idx] > 0 and intrinsic_n_total > 0:
            n_subgroup_counts.append(workgroup[d] // (n_tile_sizes[idx] * intrinsic_n_total))
        else:
            n_subgroup_counts.append(0)

    return TilingInfo(
        workgroup=workgroup,
        subgroup=subgroup,
        reduction=reduction,
        m_subgroup_counts=m_subgroup_counts,
        n_subgroup_counts=n_subgroup_counts,
        m_tile_sizes=m_tile_sizes,
        n_tile_sizes=n_tile_sizes,
        k_tile_sizes=k_tile_sizes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract workgroup/subgroup/reduction and schedule info from MLIR."
    )
    parser.add_argument("mlir_path", help="Path to MLIR file")
    args = parser.parse_args()

    tiling = extract_tiling_from_mlir(args.mlir_path)
    print(f"workgroup: {tiling.workgroup}")
    print(f"subgroup : {tiling.subgroup}")
    print(f"reduction: {tiling.reduction}")
    print(f"m_subgroup_counts: {tiling.m_subgroup_counts}")
    print(f"n_subgroup_counts: {tiling.n_subgroup_counts}")
    print(f"m_tile_sizes: {tiling.m_tile_sizes}")
    print(f"n_tile_sizes: {tiling.n_tile_sizes}")
    print(f"k_tile_sizes: {tiling.k_tile_sizes}")


if __name__ == "__main__":
    main()
