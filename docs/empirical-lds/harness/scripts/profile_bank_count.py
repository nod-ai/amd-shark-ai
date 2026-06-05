#!/usr/bin/env python3
"""Run build/lds_phase_mask and print LDS bank counters plus timer values."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import profile_common


DEFAULT_GUESSES = [1, 2, 4, 8, 16, 32, 64, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile build/lds_phase_mask for candidate LDS bank-count "
            "guesses, collecting PMC counters and benchmark timer values."
        ),
    )
    sweep_group = parser.add_argument_group("bank-count options")
    profile_common.add_access_width_argument(sweep_group, default=32)
    sweep_group.add_argument(
        "--guesses",
        nargs="+",
        type=int,
        help="Explicit bank-count guesses to test.",
    )
    sweep_group.add_argument(
        "--start",
        type=int,
        help="Inclusive range start for guesses.",
    )
    sweep_group.add_argument(
        "--stop",
        type=int,
        help="Inclusive range stop for guesses.",
    )
    sweep_group.add_argument(
        "--step",
        type=int,
        default=1,
        help="Range step for guesses.",
    )

    display_group = parser.add_argument_group("display options")
    display_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the per-guess counter/timer table.",
    )

    classifier_group = parser.add_argument_group("classification options")
    classifier_group.add_argument(
        "--min-bucket-gap",
        type=float,
        default=25.0,
        help="Warn if the largest high/low metric bucket gap is below this value.",
    )
    classifier_group.add_argument(
        "--min-bucket-gap-ratio",
        type=float,
        default=0.02,
        help="Warn if largest_gap / low_bucket_max is below this ratio.",
    )
    classifier_group.add_argument(
        "--min-gap-margin",
        type=float,
        default=10.0,
        help="Warn if largest_gap - second_largest_gap is below this many metric units.",
    )
    classifier_group.add_argument(
        "--min-gap-margin-ratio",
        type=float,
        default=0.25,
        help="Warn if (largest_gap - second_largest_gap) / largest_gap is below this ratio.",
    )
    profile_common.add_common_profiler_arguments(parser)
    return parser.parse_args()


class BankCountProfiler(profile_common.Profiler):
    def __init__(
        self,
        *,
        access_width: int,
        guesses: list[int],
        collect_pmc: bool,
        min_bucket_gap: float,
        min_bucket_gap_ratio: float,
        min_gap_margin: float,
        min_gap_margin_ratio: float,
        **profiler_args: Any,
    ) -> None:
        super().__init__(**profiler_args)
        profile_common.validate_access_width(access_width)
        self.access_width = access_width
        self.guesses = guesses
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
        self.min_bucket_gap = min_bucket_gap
        self.min_bucket_gap_ratio = min_bucket_gap_ratio
        self.min_gap_margin = min_gap_margin
        self.min_gap_margin_ratio = min_gap_margin_ratio
        self.__validate_guesses()
        profile_common.validate_nonnegative_fields(
            self,
            (
                "min_bucket_gap",
                "min_bucket_gap_ratio",
                "min_gap_margin",
                "min_gap_margin_ratio",
            ),
        )

    @staticmethod
    def choose_guesses(
        explicit_guesses: list[int] | None,
        start: int | None,
        stop: int | None,
        step: int,
    ) -> list[int]:
        using_range = start is not None or stop is not None
        if explicit_guesses is not None and using_range:
            raise profile_common.HarnessError(
                "use either --guesses or --start/--stop, not both"
            )

        if explicit_guesses is not None:
            guesses = explicit_guesses
        elif using_range:
            if start is None or stop is None:
                raise profile_common.HarnessError(
                    "--start and --stop must be provided together"
                )
            if step <= 0:
                raise profile_common.HarnessError("--step must be positive")
            guesses = list(range(start, stop + 1, step))
        else:
            guesses = DEFAULT_GUESSES

        if not guesses:
            raise profile_common.HarnessError("no guesses selected")
        return guesses

    def __validate_guesses(self) -> None:
        for guess in self.guesses:
            if guess <= 0:
                raise profile_common.HarnessError(f"guess must be positive: {guess}")

    def __profile_guess(self, guess: int) -> tuple[dict[str, object], list[str]]:
        average_record, run_warnings = self.profile_phase_mask_timer_observation(
            label=guess,
            run_prefix=f"w{self.access_width}_guess_{guess}",
            access_width=self.access_width,
            thread_mask=profile_common.full_thread_mask(self.wavefront_threads),
            bank_count=guess,
            threads=self.wavefront_threads,
            error_context=(
                f"rocprofv3 counters/timer for width={self.access_width} "
                f"guess={guess}"
            ),
            pmc_counter=self.lds_bank_conflict_counter,
        )
        cycles_avg = average_record["cycles_avg"]
        if not isinstance(cycles_avg, float):
            raise profile_common.HarnessError(
                f"unexpected averaged cycles_avg for guess={guess}: {cycles_avg!r}"
            )
        return average_record, run_warnings

    def find_bank_count(self, *, print_rows: bool = False) -> dict[str, object]:
        warnings: list[str] = list(self.startup_warnings)
        records_by_guess: dict[int, dict[str, object]] = {}

        for guess in self.guesses:
            average_record, guess_warnings = self.__profile_guess(guess)
            records_by_guess[guess] = average_record
            warnings.extend(guess_warnings)
            if print_rows:
                profile_common.print_timer_table_record(
                    average_record,
                    label_header="guess",
                )

        (
            classification_metric,
            classification_values,
        ) = profile_common.latency_metric_values(records_by_guess)
        bucket_summary, bucket_warnings = self.__classify_metric_buckets(
            classification_values,
        )
        return {
            "most_likely_bank_count": int(bucket_summary["most_likely_bank_count"]),
            "classification_metric": classification_metric,
            "warnings": profile_common.unique_warnings(warnings),
            "bucket_summary": bucket_summary,
            "bucket_warnings": bucket_warnings,
        }

    def __classify_metric_buckets(
        self,
        metric_values_by_guess: dict[int, float],
    ) -> tuple[dict[str, object], list[str]]:
        if not metric_values_by_guess:
            raise profile_common.HarnessError(
                "no classification metric values were found"
            )

        sorted_values = sorted(
            metric_values_by_guess.items(),
            key=lambda item: (item[1], item[0]),
        )
        warnings: list[str] = []

        if len(sorted_values) == 1:
            guess, value = sorted_values[0]
            warnings.append(
                "only one distinct guess was profiled; bucket split is unavailable"
            )
            return (
                {
                    "high_bucket": {guess: value},
                    "low_bucket": {},
                    "most_likely_bank_count": guess,
                    "largest_gap": None,
                    "second_largest_gap": None,
                    "gap_margin": None,
                    "gap_ratio": None,
                    "gap_margin_ratio": None,
                },
                warnings,
            )

        gaps: list[tuple[int, float]] = []
        for idx in range(len(sorted_values) - 1):
            low_value = sorted_values[idx][1]
            high_value = sorted_values[idx + 1][1]
            gaps.append((idx, high_value - low_value))

        ranked_gaps = sorted(gaps, key=lambda item: (-item[1], item[0]))
        split_index, largest_gap = ranked_gaps[0]
        second_largest_gap = ranked_gaps[1][1] if len(ranked_gaps) > 1 else None

        if largest_gap == 0:
            low_bucket: dict[int, float] = {}
            high_bucket = dict(sorted(metric_values_by_guess.items()))
            gap_ratio = 0.0
        else:
            low_items = sorted_values[: split_index + 1]
            high_items = sorted_values[split_index + 1 :]
            low_bucket = dict(sorted(low_items))
            high_bucket = dict(sorted(high_items))
            low_bucket_max = low_items[-1][1]
            gap_ratio = largest_gap / low_bucket_max if low_bucket_max else float("inf")

        if largest_gap < self.min_bucket_gap or gap_ratio < self.min_bucket_gap_ratio:
            warnings.append(
                "weak metric bucket separation: "
                f"largest_gap={profile_common.format_value(largest_gap)} "
                f"gap_ratio={profile_common.format_value(gap_ratio)}"
            )

        gap_margin = None
        gap_margin_ratio = None
        if second_largest_gap is not None:
            gap_margin = largest_gap - second_largest_gap
            gap_margin_ratio = gap_margin / largest_gap if largest_gap else 0.0
            if (
                gap_margin < self.min_gap_margin
                or gap_margin_ratio < self.min_gap_margin_ratio
            ):
                warnings.append(
                    "ambiguous metric bucket split: "
                    f"largest_gap={profile_common.format_value(largest_gap)} "
                    f"second_largest_gap={profile_common.format_value(second_largest_gap)} "
                    f"gap_margin={profile_common.format_value(gap_margin)} "
                    f"gap_margin_ratio={profile_common.format_value(gap_margin_ratio)}"
                )

        return (
            {
                "high_bucket": high_bucket,
                "low_bucket": low_bucket,
                "most_likely_bank_count": min(high_bucket),
                "largest_gap": largest_gap,
                "second_largest_gap": second_largest_gap,
                "gap_margin": gap_margin,
                "gap_ratio": gap_ratio,
                "gap_margin_ratio": gap_margin_ratio,
            },
            warnings,
        )


def print_bucket_summary(
    summary: dict[str, object],
    warnings: list[str],
    classification_metric: object,
) -> int:
    high_bucket = summary["high_bucket"]
    assert isinstance(high_bucket, dict)
    classification_field = str(classification_metric)
    high_bucket_text = ", ".join(
        f"{guess}:{profile_common.format_value(latency, classification_field)}"
        for guess, latency in sorted(high_bucket.items())
    )
    lines = [
        f"\nclassification_metric={classification_metric}",
        f"high_metric_bucket={high_bucket_text}",
        f"largest_gap={profile_common.format_value(summary['largest_gap'], 'largest_gap')}",
        "second_largest_gap="
        f"{profile_common.format_value(summary['second_largest_gap'], 'second_largest_gap')}",
        f"gap_margin={profile_common.format_value(summary['gap_margin'], 'gap_margin')}",
        f"gap_ratio={profile_common.format_value(summary['gap_ratio'], 'gap_ratio')}",
        "gap_margin_ratio="
        f"{profile_common.format_value(summary['gap_margin_ratio'], 'gap_margin_ratio')}",
    ]
    for warning in warnings:
        lines.append(f"warning: {warning}")
    lines.append(f"most_likely_bank_count={summary['most_likely_bank_count']}")
    print("\n".join(lines), flush=True)
    return int(summary["most_likely_bank_count"])


def profiler_from_args(args: argparse.Namespace) -> BankCountProfiler:
    root = profile_common.repo_root()
    guesses = BankCountProfiler.choose_guesses(
        args.guesses,
        args.start,
        args.stop,
        args.step,
    )
    return BankCountProfiler(
        **profile_common.profiler_kwargs_from_args(
            args,
            root=root,
            access_width=args.access_width,
        ),
        access_width=args.access_width,
        guesses=guesses,
        collect_pmc=not args.no_pmc,
        min_bucket_gap=args.min_bucket_gap,
        min_bucket_gap_ratio=args.min_bucket_gap_ratio,
        min_gap_margin=args.min_gap_margin,
        min_gap_margin_ratio=args.min_gap_margin_ratio,
    )


def print_bank_count_result(result: dict[str, object]) -> None:
    warnings = result["warnings"]
    bucket_summary = result["bucket_summary"]
    bucket_warnings = result["bucket_warnings"]
    classification_metric = result["classification_metric"]

    assert isinstance(warnings, list)
    assert isinstance(bucket_summary, dict)
    assert isinstance(bucket_warnings, list)

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    print_bucket_summary(bucket_summary, bucket_warnings, classification_metric)


def run_from_args(args: argparse.Namespace) -> int:
    try:
        profiler = profiler_from_args(args)
        if args.verbose:
            profile_common.print_timer_table_header(label_header="guess")
        result = profiler.find_bank_count(print_rows=args.verbose)
        print_bank_count_result(result)
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
