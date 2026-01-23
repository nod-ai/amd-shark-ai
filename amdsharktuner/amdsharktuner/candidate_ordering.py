import random
import logging
import csv
import math
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum
from typing import Optional, Callable

from iree.compiler.dialects import iree_gpu  # type: ignore

from . import common


class CandidateOrderKind(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"


def is_pow2(x: int) -> bool:
    # Return True if is power of 2.
    return x > 0 and (x & (x - 1)) == 0


def is_mult_simd_num(x: int, simd_num: int) -> bool:
    # Return True if is a multiple of 4 (number of SIMDs in a CU).
    return x % simd_num == 0


def arith_intensity(x: int, y: int, z: int) -> float:
    num_flops = 2 * x * y * z
    num_byte_access = 2 * (x * y + y * z + x * z)
    return num_flops / num_byte_access


def quantization_inefficiency(problem_m, tile_m, problem_n, tile_n, cu_num) -> float:
    # Inefficiency of tiling when problem sizes do not divide evenly,
    # resulting in wasted computation in the final tiling round.
    num_workgroups = (problem_m / tile_m) * (problem_n / tile_n)
    ceil_val = math.ceil(num_workgroups / cu_num)
    q_ie = (ceil_val - num_workgroups / cu_num) / ceil_val
    return q_ie


def size_ratio(x: int, y: int) -> float:
    return min(x, y) / max(x, y)


# Generic sort key map - architecture-specific modules can extend this.
SORT_KEY_MAP: dict[type[common.KnobAssignment | None], Callable | None] = {
    type(None): None,
}


def reorder_assignments(
    knobs: list[Optional[common.KnobAssignment]],
    strategy: CandidateOrderKind,
    key_fn: Optional[Callable] = None,
    target_info: Optional[iree_gpu.TargetInfo] = None,
    sort_key_map: Optional[
        dict[type[common.KnobAssignment | None], Callable | None]
    ] = None,
) -> list[int]:
    """
    Returns a list of indices representing the new order relative to the original list.
    Example: ['a', 'b', 'c'] -> ['b', 'a', 'c'], return [1, 0, 2]
    """
    logging.debug(f"Selected candidate ordering strategy: {strategy}")

    if not knobs:
        return []

    original_order = list(range(len(knobs)))  # Identity mapping.

    key_fn_to_use: Optional[Callable] = None
    match strategy:
        case CandidateOrderKind.no_sort:
            return original_order
        case CandidateOrderKind.shuffle:
            indices = list(range(len(knobs)))
            random.shuffle(indices)
            return indices
        case CandidateOrderKind.heuristic:
            # Auto set a sort key function based on the knob type.
            knob_type = type(knobs[0])
            key_fn_to_use = (
                key_fn if key_fn else (sort_key_map or SORT_KEY_MAP).get(knob_type)
            )
            if key_fn_to_use is None:
                logging.warning(
                    f"No sort key defined for knob type {knob_type.__name__}."
                )
                return original_order
            logging.debug(f"Selected sort key: {key_fn_to_use.__name__}")

            indexed_list = list(enumerate(knobs))
            # Good candidates are sorted to the front of the list.
            if not key_fn:
                # If no custom sort key is specified, use the key selected from SORT_KEY_MAP.
                assert (
                    target_info
                ), "The selected heuristic reordering function requires target information to be provided"
                sorted_list = sorted(
                    indexed_list, key=lambda pair: key_fn_to_use(pair[1], target_info)
                )
            else:
                sorted_list = sorted(
                    indexed_list, key=lambda pair: key_fn_to_use(pair[1])
                )
            logging.warning(f"Heuristic candidate reordering applied.")
            indices = [i for i, _ in sorted_list]
            return indices
        case _:
            assert False


@dataclass
class TuningRecord:
    """
    Records a candidate's knob configuration and tuning results.

    Used to analyze the candidate search space and to evaluate the
    effectiveness of candidate ordering heuristics.
    """

    gen_id: int  # Original index from candidate generation.
    candidate_id: int  # Index in candidate_trackers after reordering.
    knob: Optional[common.KnobAssignment] = None
    to_compile: bool = False
    compile_status: bool = False
    to_benchmark: bool = False
    benchmark_device_id: Optional[str] = None
    benchmark_queue_position: Optional[int] = None
    benchmark_status: bool = False
    baseline_benchmark_time_us: Optional[float] = None
    benchmark_time_us: Optional[float] = None
    benchmark_speedup: Optional[float] = None
    benchmark_rank_order: Optional[int] = None


def build_tuning_records_from_order(
    knobs: list[Optional[common.KnobAssignment]], sorted_order: list[int]
) -> list[TuningRecord]:
    tuning_records: list[TuningRecord] = []
    # Insert baseline entry (always candidate_id = 0, gen_id = 0).
    tuning_records.append(TuningRecord(gen_id=0, candidate_id=0, knob=None))
    for sorted_position, original_gen_index in enumerate(sorted_order, start=1):
        tr = TuningRecord(
            gen_id=original_gen_index
            + 1,  # Shift by 1 to reserve gen_id=0 for baseline.
            candidate_id=sorted_position,
            knob=knobs[original_gen_index],
        )
        tuning_records.append(tr)

    return tuning_records


def flatten_records(
    tuning_records: list[TuningRecord],
) -> list[dict[str, Any]]:
    """
    Flatten a list of `TuningRecord` objects into CSV headers and rows.

    - Each record becomes one CSV row.
    - Top-level attributes (e.g., `gen_id`, `benchmark_time_us`) appear as individual columns.
    - Nested objects (e.g., `knob`) are flattened into columns like `knob.M`, `knob.tile_m`.
    """
    rows = []
    for tuning_record in tuning_records:
        # Drop the baseline entry due to missing knob info.
        if not tuning_record.knob:
            continue
        row = {}
        for attr, val in vars(tuning_record).items():
            if isinstance(val, common.KnobAssignment):
                knob_dict = val.get_knobs()
                for k, v in knob_dict.items():
                    row[f"{attr}_{k}"] = v
            else:
                row[attr] = val
        rows.append(row)

    return rows


def export_record_to_csv(tuning_records: list[TuningRecord], dest_file: Path) -> None:
    assert tuning_records

    rows = flatten_records(tuning_records)
    headers = list(rows[0].keys())

    with open(dest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
