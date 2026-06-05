#!/usr/bin/env python3
"""Run build/lds_phase_mask for mask patterns."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import Any, Callable, Iterable

import profile_common


MaskPattern = tuple[str, int]
MaskPatternGenerator = Callable[[int], Iterable[MaskPattern]]


class MaskPatternGenerators:
    @staticmethod
    def contiguous_mask(lane_count: int) -> int:
        if lane_count <= 0 or lane_count > 64:
            raise profile_common.HarnessError(
                f"mask lane count must be in [1, 64]: {lane_count}"
            )
        return (1 << lane_count) - 1

    @classmethod
    def halving(cls, wavefront_threads: int) -> Iterable[MaskPattern]:
        yield "all", cls.contiguous_mask(wavefront_threads)
        lane_count = wavefront_threads // 2
        while lane_count >= 1:
            label = "0" if lane_count == 1 else f"0-{lane_count - 1}"
            yield label, cls.contiguous_mask(lane_count)
            lane_count //= 2
        yield "none", 0

    @classmethod
    def shift_full(cls, wavefront_threads: int) -> Iterable[MaskPattern]:
        full_mask = cls.contiguous_mask(wavefront_threads)
        for shift in range(wavefront_threads + 1):
            mask = (full_mask << shift) & full_mask
            if shift == 0:
                label = "all"
            elif shift == wavefront_threads:
                label = "none"
            elif shift == wavefront_threads - 1:
                label = str(shift)
            else:
                label = f"{shift}-{wavefront_threads - 1}"
            yield label, mask

    @staticmethod
    def lane0_pairs(wavefront_threads: int) -> Iterable[MaskPattern]:
        for lane in range(1, wavefront_threads):
            yield f"0,{lane}", (1 << 0) | (1 << lane)

    @classmethod
    def window(cls, wavefront_threads: int, window: int) -> Iterable[MaskPattern]:
        if window <= 0:
            raise profile_common.HarnessError(f"window size must be positive: {window}")
        if window > wavefront_threads:
            raise profile_common.HarnessError(
                f"window size {window} exceeds wavefront size {wavefront_threads}"
            )
        for start in range(wavefront_threads - window + 1):
            stop = start + window - 1
            yield f"{start}-{stop}", ((1 << window) - 1) << start

    @classmethod
    def window_generator(cls, window: int) -> MaskPatternGenerator:
        def generate(wavefront_threads: int) -> Iterable[MaskPattern]:
            return cls.window(wavefront_threads, window)

        return generate


MASK_PATTERN_GENERATORS: dict[str, MaskPatternGenerator] = {
    "lane0-pairs": MaskPatternGenerators.lane0_pairs,
    "halving": MaskPatternGenerators.halving,
    "shift-full": MaskPatternGenerators.shift_full,
    **{
        f"window-{window}": MaskPatternGenerators.window_generator(window)
        for window in range(1, 11)
    },
    "window-16": MaskPatternGenerators.window_generator(16),
}
DEFAULT_PATTERN = "lane0-pairs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile build/lds_phase_mask for thread-mask patterns, "
            "collecting PMC counters and benchmark timer values."
        ),
    )
    mask_group = parser.add_argument_group("mask-pattern options")
    profile_common.add_access_width_argument(mask_group, default=64)
    mask_group.add_argument(
        "--bank-count",
        type=int,
        default=32,
        help="Number of memory banks.",
    )
    mask_group.add_argument(
        "--pattern",
        choices=sorted(MASK_PATTERN_GENERATORS),
        default=DEFAULT_PATTERN,
        help="Mask pattern generator to run.",
    )
    profile_common.add_common_profiler_arguments(parser)
    return parser.parse_args()


class PhaseExperimentProfiler(profile_common.Profiler):
    def __init__(
        self,
        *,
        access_width: int,
        bank_count: int,
        pattern_generator: MaskPatternGenerator,
        collect_pmc: bool,
        **profiler_args: Any,
    ) -> None:
        super().__init__(**profiler_args)
        profile_common.validate_access_width(access_width)
        if bank_count <= 0:
            raise profile_common.HarnessError("--bank-count must be positive")

        self.access_width = access_width
        self.bank_count = bank_count
        self.pattern_generator = pattern_generator
        self.lds_bank_conflict_counter: str | None = None
        self.startup_warnings: list[str] = []
        if collect_pmc:
            (
                self.lds_bank_conflict_counter,
                counter_warning,
            ) = self.choose_optional_lds_bank_conflict_counter()
            if counter_warning is not None:
                self.startup_warnings.append(counter_warning)
        self.wavefront_threads = self.query_thread_mask_wavefront_threads()

    @staticmethod
    def table_columns() -> list[profile_common.TableColumn]:
        return [
            ("pattern", "mask", 8),
            ("mask_hex", "mask_hex", 18),
            *profile_common.Profiler.timer_table_columns()[1:],
        ]

    def __profile_mask(
        self, label: str, mask: int
    ) -> tuple[dict[str, object], list[str]]:
        average_record, run_warnings = self.profile_phase_mask_timer_observation(
            label=label,
            run_prefix="mask_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_"),
            access_width=self.access_width,
            thread_mask=mask,
            bank_count=self.bank_count,
            threads=self.wavefront_threads,
            error_context=f"rocprofv3 counters/timer for mask={label}",
            pmc_counter=self.lds_bank_conflict_counter,
        )
        cycles_avg = average_record["cycles_avg"]
        if not isinstance(cycles_avg, float):
            raise profile_common.HarnessError(
                f"unexpected averaged cycles_avg for mask={label}: {cycles_avg!r}"
            )
        average_record["pattern"] = label
        average_record["mask_hex"] = profile_common.format_thread_mask(mask)
        return average_record, run_warnings

    def profile_masks(self, *, print_rows: bool = False) -> list[str]:
        warnings: list[str] = []
        columns = self.table_columns()
        for label, mask in self.pattern_generator(self.wavefront_threads):
            record, mask_warnings = self.__profile_mask(label, mask)
            warnings.extend(mask_warnings)
            if print_rows:
                profile_common.print_table_record(record, columns)
        return profile_common.unique_warnings(warnings)


def profiler_from_args(args: argparse.Namespace) -> PhaseExperimentProfiler:
    root = profile_common.repo_root()
    return PhaseExperimentProfiler(
        **profile_common.profiler_kwargs_from_args(
            args,
            root=root,
            access_width=args.access_width,
        ),
        access_width=args.access_width,
        bank_count=args.bank_count,
        pattern_generator=MASK_PATTERN_GENERATORS[args.pattern],
        collect_pmc=not args.no_pmc,
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        profiler = profiler_from_args(args)
        for warning in profiler.startup_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        profile_common.print_table_header(PhaseExperimentProfiler.table_columns())
        run_warnings = profiler.profile_masks(print_rows=True)
        for warning in run_warnings:
            print(f"warning: {warning}", file=sys.stderr)
    except (profile_common.HarnessError, OSError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    parsed_args = parse_args()
    raise SystemExit(run_from_args(parsed_args))
