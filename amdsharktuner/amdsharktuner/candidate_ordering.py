import random
import logging
import csv
from collections.abc import Sequence
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from iree.compiler.dialects import iree_gpu  # type: ignore

from . import common
from .rocm import rocm_candidate_ordering


class CandidateOrderKind(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"

    def __str__(self) -> str:
        return self.value


def reorder_solutions(
    solutions: list[common.SMTKnobAssignments],
    strategy: CandidateOrderKind,
    codegen_pipeline: iree_gpu.LoweringPipeline,
    dispatch_kind: common.DispatchKind,
) -> list[int]:
    """
    Returns a list of indices representing the new order relative to the original list.
    Example: ['a', 'b', 'c'] -> ['b', 'a', 'c'], return [1, 0, 2]
    """
    logging.debug(f"Selected candidate ordering strategy: {strategy}")

    if not solutions:
        return []

    original_order = list(range(len(solutions)))

    match strategy:
        case CandidateOrderKind.no_sort:
            return original_order
        case CandidateOrderKind.shuffle:
            indices = list(range(len(solutions)))
            random.shuffle(indices)
            return indices
        case CandidateOrderKind.heuristic:
            match codegen_pipeline:
                case (
                    iree_gpu.LoweringPipeline.VectorDistribute
                    | iree_gpu.LoweringPipeline.TileAndFuse
                ):
                    heuristic_key_fn = rocm_candidate_ordering.get_heuristic_key_fn(
                        solutions, codegen_pipeline, dispatch_kind
                    )
                case _:
                    logging.warning(
                        f"No heuristic candidate ordering is defined for "
                        f"{codegen_pipeline.name}; using the original order. "
                        "Supported pipelines: VectorDistribute, TileAndFuse."
                    )
                    return original_order
            if heuristic_key_fn is None:
                return original_order

            sorted_list = sorted(original_order, key=heuristic_key_fn)
            logging.info("Heuristic candidate reordering applied.")
            return sorted_list
        case _:
            assert False


@dataclass
class TuningRecord:
    """
    Records a candidate's knob configuration and tuning results.

    Used to analyze the candidate search space and to evaluate the
    results from compile and benchmark phases.
    """

    gen_id: int  # Original index from candidate generation.
    candidate_id: int  # Index in candidate_trackers after reordering.
    solution: Optional[common.SMTKnobAssignments] = None
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
    solutions: Sequence[common.SMTKnobAssignments],
    sorted_order: list[int],
) -> list[TuningRecord]:
    tuning_records: list[TuningRecord] = []
    # Insert baseline entry (always candidate_id = 0, gen_id = 0).
    tuning_records.append(TuningRecord(gen_id=0, candidate_id=0))
    for sorted_position, original_gen_index in enumerate(sorted_order, start=1):
        tr = TuningRecord(
            gen_id=original_gen_index + 1,
            candidate_id=sorted_position,
            solution=solutions[original_gen_index],
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
    - Nested solution dictionaries are flattened into columns like `solution_wg_0`.
    """
    rows = []
    for tuning_record in tuning_records:
        # Drop only the baseline entry. Some dispatches do not currently expose
        # recognized knob metadata, but their compile/benchmark results are
        # still useful and should be exported.
        if tuning_record.candidate_id == 0:
            continue
        row = {}
        for attr, val in vars(tuning_record).items():
            if isinstance(val, dict):
                for k, v in val.items():
                    row[f"{attr}_{k}"] = v
            else:
                row[attr] = val
        rows.append(row)

    return rows


def prepare_record_csv_data(
    tuning_records: list[TuningRecord],
) -> tuple[list[str], list[dict[str, Any]]]:
    rows = flatten_records(tuning_records)
    if not rows:
        return [], []

    return list(rows[0].keys()), rows


def export_record_to_csv(tuning_records: list[TuningRecord], dest_file: Path) -> None:
    assert tuning_records

    headers, rows = prepare_record_csv_data(tuning_records)
    if not rows:
        logging.warning(f"No candidate tuning records to export to {dest_file}.")
        dest_file.write_text("", encoding="utf-8")
        return

    with open(dest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
