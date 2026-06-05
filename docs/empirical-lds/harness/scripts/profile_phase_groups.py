#!/usr/bin/env python3
"""Run the phase-mask benchmark and print phase groups."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import profile_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile build/lds_phase_mask for phase-group pair comparisons, "
            "collecting PMC counters and benchmark timer values."
        ),
    )
    phase_group = parser.add_argument_group("phase-group options")
    profile_common.add_access_width_argument(phase_group, required=True)
    phase_group.add_argument(
        "--bank-count",
        type=int,
        default=32,
        help="Number of memory banks.",
    )
    display_group = parser.add_argument_group("display options")
    display_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print comparison counter/timer rows and group progress.",
    )
    profile_common.add_common_profiler_arguments(parser)
    return parser.parse_args()


class PhaseGroupProfiler(profile_common.Profiler):
    def __init__(
        self,
        *,
        access_width: int,
        bank_count: int,
        print_progress: bool,
        collect_pmc: bool,
        **profiler_args: Any,
    ) -> None:
        super().__init__(**profiler_args)
        profile_common.validate_access_width(access_width)
        if bank_count <= 0:
            raise profile_common.HarnessError("--bank-count must be positive")

        phase_group_size = 32 * bank_count // access_width
        if phase_group_size < 1:
            raise profile_common.HarnessError(
                "computed phase group size is zero; "
                f"bank_count={bank_count} access_width={access_width}"
            )

        self.access_width = access_width
        self.bank_count = bank_count
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
        self.print_progress = print_progress
        self.phase_group_size = phase_group_size
        self.phase_group_count = (
            self.wavefront_threads + phase_group_size - 1
        ) // phase_group_size

    def __query_observation(
        self,
        t1: int,
        t2: int,
        bank_count: int,
    ) -> tuple[dict[str, object], list[str]]:
        thread_mask = (1 << t1) | (1 << t2)
        average_record, run_warnings = self.profile_phase_mask_timer_observation(
            label=f"t{t1}t{t2}",
            run_prefix=f"w{self.access_width}_compare_t{t1}t{t2}",
            access_width=self.access_width,
            thread_mask=thread_mask,
            bank_count=bank_count,
            threads=self.wavefront_threads,
            error_context=(
                f"rocprofv3 counters/timer for width={self.access_width} " f"t{t1}t{t2}"
            ),
            pmc_counter=self.lds_bank_conflict_counter,
        )
        cycles_avg = average_record["cycles_avg"]
        if not isinstance(cycles_avg, float):
            raise profile_common.HarnessError(
                f"unexpected averaged cycles_avg for label=t{t1}t{t2}: {cycles_avg!r}"
            )
        return average_record, run_warnings

    def find_phase_groups(self) -> tuple[list[int], list[str], str]:
        # The group each thread belongs to
        threads = self.wavefront_threads
        tgroup = [-1 for i in range(threads)]
        gcurr = 0
        run_warnings: list[str] = []
        classification_metric = "none"
        for t1 in range(threads):
            if tgroup[t1] != -1:
                continue

            # Previous passes assign exactly one full phase group. Once only
            # one group remains, the remaining lanes are determined by
            # elimination and do not need another noisy classifier pass.
            if gcurr == self.phase_group_count - 1:
                for thread, group in enumerate(tgroup):
                    if group == -1:
                        tgroup[thread] = gcurr
                if self.print_progress:
                    members = [
                        thread for thread, group in enumerate(tgroup) if group == gcurr
                    ]
                    self.__print_phase_group_members(gcurr, members)
                break

            tgroup[t1] = gcurr
            gcount = 1
            records_by_thread: dict[int, dict[str, object]] = {}
            for t2 in range(t1 + 1, threads):
                if tgroup[t2] != -1:
                    continue
                record, warnings = self.__query_observation(t1, t2, self.bank_count)
                records_by_thread[t2] = record
                run_warnings.extend(warnings)
                if self.print_progress:
                    profile_common.print_timer_table_record(record)
            # Collect the threads with the highest selected metric.
            if not records_by_thread:
                if gcount < self.phase_group_size:
                    run_warnings.append(
                        f"phase group {gcurr} had no candidate observations; "
                        f"assigned {gcount} of {self.phase_group_size} expected lanes"
                    )
                if self.print_progress:
                    members = [
                        thread for thread, group in enumerate(tgroup) if group == gcurr
                    ]
                    self.__print_phase_group_members(gcurr, members)
                gcurr += 1
                continue

            (
                current_metric,
                classification_values,
            ) = profile_common.latency_metric_values(records_by_thread)
            if classification_metric == "none":
                classification_metric = current_metric
            elif classification_metric != current_metric:
                run_warnings.append(
                    "classification metric changed across phase-group passes: "
                    f"{classification_metric} then {current_metric}"
                )
            for t, value in sorted(
                classification_values.items(),
                reverse=True,
                key=lambda item: item[1],
            ):
                if gcount == self.phase_group_size:
                    break
                tgroup[t] = gcurr
                gcount += 1
            if self.print_progress:
                members = [
                    thread for thread, group in enumerate(tgroup) if group == gcurr
                ]
                self.__print_phase_group_members(gcurr, members)
            gcurr += 1

        return (
            tgroup,
            profile_common.unique_warnings(run_warnings),
            classification_metric,
        )

    @staticmethod
    def __print_phase_group_members(group: int, members: list[int]) -> None:
        member_text = PhaseGroupProfiler.__format_member_ranges(members)
        print(f"phase_group {group}: {member_text}", flush=True)

    @staticmethod
    def __format_member_ranges(members: list[int]) -> str:
        sorted_members = sorted(set(members))
        if not sorted_members:
            return ""

        ranges: list[tuple[int, int]] = []
        start = sorted_members[0]
        end = sorted_members[0]
        for member in sorted_members[1:]:
            if member == end + 1:
                end = member
                continue
            ranges.append((start, end))
            start = member
            end = member
        ranges.append((start, end))

        return ", ".join(
            str(start) if start == end else f"{start}-{end}" for start, end in ranges
        )

    @staticmethod
    def print_phase_groups(tgroup: list[int]) -> None:
        for group in sorted(set(tgroup)):
            members = [
                thread
                for thread, thread_group in enumerate(tgroup)
                if thread_group == group
            ]
            PhaseGroupProfiler.__print_phase_group_members(group, members)


def profiler_from_args(args: argparse.Namespace) -> PhaseGroupProfiler:
    root = profile_common.repo_root()
    return PhaseGroupProfiler(
        **profile_common.profiler_kwargs_from_args(
            args,
            root=root,
            access_width=args.access_width,
        ),
        access_width=args.access_width,
        bank_count=args.bank_count,
        print_progress=args.verbose,
        collect_pmc=not args.no_pmc,
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        profiler = profiler_from_args(args)
        for warning in profiler.startup_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        if profiler.print_progress:
            profile_common.print_timer_table_header()
        tgroup, run_warnings, classification_metric = profiler.find_phase_groups()
        print(f"classification_metric={classification_metric}")
        profiler.print_phase_groups(tgroup)
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
