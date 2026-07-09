"""Shared helpers for rocprof profiling scripts."""

from __future__ import annotations

import csv
import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable, TypeVar


TIMER_FIELDS = ("cycles_avg", "min_thread_latency", "max_thread_latency")
RAW_LDS_BANK_CONFLICT_COUNTERS = ("SQC_LDS_BANK_CONFLICT", "SQ_LDS_BANK_CONFLICT")
COUNTER_COLLECTION_REQUIRED_COLUMNS = {"Counter_Name", "Counter_Value"}
COUNTER_NAME_PATTERN = re.compile(r"^\s*Counter_Name\s*:\s*(\S+)\s*$", re.MULTILINE)
PHASE_MASK_EXECUTABLE = "lds_phase_mask"
PHASE_MASK_WORKGROUP_COUNT = 4096
SUPPORTED_ACCESS_WIDTHS = (32, 64, 128)

TableColumn = tuple[str, str, int]
ClassificationKey = TypeVar("ClassificationKey")


class HarnessError(RuntimeError):
    pass


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def tool_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parent == Path(".") and shutil.which(str(path)):
        return path
    return root / path


def access_width_choices() -> list[int]:
    return list(SUPPORTED_ACCESS_WIDTHS)


def validate_access_width(access_width: int) -> None:
    if access_width not in SUPPORTED_ACCESS_WIDTHS:
        raise HarnessError(
            f"unsupported access width {access_width}; "
            f"expected one of {access_width_choices()}"
        )


def format_thread_mask(mask: int) -> str:
    return f"0x{mask:016x}"


def full_thread_mask(wavefront_threads: int) -> int:
    if wavefront_threads <= 0:
        raise HarnessError(
            f"wavefront thread count must be positive: {wavefront_threads}"
        )
    if wavefront_threads > 64:
        raise HarnessError(
            "lds_phase_mask uses a 64-bit thread mask, but the benchmark "
            f"reported {wavefront_threads} wavefront threads"
        )
    return (1 << wavefront_threads) - 1


def phase_mask_stdin(
    *,
    access_width: int,
    thread_mask: int,
    bank_count: int,
    threads: int,
) -> str:
    return f"{access_width} {format_thread_mask(thread_mask)} {bank_count} {threads}\n"


def add_access_width_argument(
    group: argparse._ArgumentGroup,
    *,
    required: bool = False,
    default: int | None = None,
) -> None:
    group.add_argument(
        "--access-width",
        type=int,
        required=required,
        default=default,
        choices=access_width_choices(),
        help="LDS access width to profile, in bits.",
    )


def add_common_profiler_arguments(parser: argparse.ArgumentParser) -> None:
    root = repo_root()
    benchmark_group = parser.add_argument_group("benchmark options")
    benchmark_group.add_argument(
        "--executable",
        default=str(root / "build" / PHASE_MASK_EXECUTABLE),
        help=f"Path to the {PHASE_MASK_EXECUTABLE} executable.",
    )

    profiler_group = parser.add_argument_group("profiler options")
    profiler_group.add_argument(
        "--rocprof",
        default=str(default_rocprof(root)),
        help="Path to rocprofv3. Defaults to the project venv.",
    )
    profiler_group.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per observation.",
    )
    profiler_group.add_argument(
        "--kernel-regex",
        help="Only keep kernels matching this regex. Also passed to rocprof.",
    )
    profiler_group.add_argument(
        "--rocprof-verbose",
        action="store_true",
        help="Print rocprof commands, stdout, and stderr.",
    )
    profiler_group.add_argument(
        "--no-pmc",
        action="store_true",
        help="Do not request or print the LDS bank-conflict PMC counter.",
    )

    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "--output-root",
        default="rocprof",
        help="Directory for raw rocprof output.",
    )


def profiler_kwargs_from_args(
    args: argparse.Namespace,
    *,
    root: Path,
    access_width: int,
) -> dict[str, object]:
    validate_access_width(access_width)
    rocprof = None if args.no_pmc else tool_path(args.rocprof, root)
    return {
        "root": root,
        "rocprof": rocprof,
        "executable": tool_path(args.executable, root),
        "output_root": Path(args.output_root),
        "runs": args.runs,
        "kernel_regex": args.kernel_regex,
        "verbose": args.rocprof_verbose,
    }


def prepend_env_path(env: dict[str, str], name: str, paths: Iterable[Path]) -> None:
    existing = env.get(name, "")
    prefix = [str(path) for path in paths if path.exists()]
    if not prefix:
        return
    env[name] = os.pathsep.join(prefix + ([existing] if existing else []))


def rocm_root_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(
        sorted((root / "venv/lib").glob("python*/site-packages/_rocm_sdk_devel"))
    )
    candidates.extend(
        sorted((root / "venv/lib").glob("python*/site-packages/_rocm_sdk_core"))
    )
    for env_name in ("ROCM_PATH", "ROCM_SDK_ROOT", "HIP_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    candidates.append(Path("/opt/rocm"))

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def infer_rocm_root(root: Path) -> Path | None:
    for candidate in rocm_root_candidates(root):
        if candidate.exists():
            return candidate
    return None


def default_rocprof(root: Path) -> Path:
    direct = root / "venv/bin/rocprofv3"
    if direct.exists():
        return direct

    rocm_root = infer_rocm_root(root)
    if rocm_root is not None:
        rocm_rocprof = rocm_root / "bin/rocprofv3"
        if rocm_rocprof.exists():
            return rocm_rocprof

    for rocm_candidate in rocm_root_candidates(root):
        rocm_rocprof = rocm_candidate / "bin/rocprofv3"
        if rocm_rocprof.exists():
            return rocm_rocprof

    path_rocprof = shutil.which("rocprofv3")
    if path_rocprof:
        return Path(path_rocprof)

    return Path("rocprofv3")


def rocprof_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    rocm_root = infer_rocm_root(root)
    if rocm_root is None:
        return env

    env.setdefault("ROCM_PATH", str(rocm_root))
    env.setdefault("ROCM_SDK_ROOT", str(rocm_root))
    env.setdefault("HIP_PATH", str(rocm_root))
    env.setdefault("HIP_PLATFORM", "amd")
    env.setdefault("HIP_COMPILER", "clang")
    env.setdefault("HIP_RUNTIME", "rocclr")

    device_lib_path = rocm_root / "lib/llvm/amdgcn/bitcode"
    if device_lib_path.exists():
        env.setdefault("HIP_DEVICE_LIB_PATH", str(device_lib_path))
        env.setdefault("DEVICE_LIB_PATH", str(device_lib_path))

    metrics_path = rocm_root / "share/rocprofiler-sdk"
    if metrics_path.exists():
        env.setdefault("ROCPROFILER_METRICS_PATH", str(metrics_path))

    prepend_env_path(env, "PATH", [rocm_root / "bin", rocm_root / "lib/llvm/bin"])
    prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [
            rocm_root / "lib",
            rocm_root / "lib/llvm/lib",
            rocm_root / "lib/rocprofiler-sdk",
        ],
    )
    return env


def validate_executable(path: Path, label: str) -> None:
    if not path.is_absolute() and shutil.which(str(path)):
        return
    if not path.exists():
        raise HarnessError(f"{label} not found: {path}")
    if not os.access(path, os.X_OK):
        raise HarnessError(f"{label} is not executable: {path}")


def validate_nonnegative_fields(obj: object, names: Iterable[str]) -> None:
    for name in names:
        value = getattr(obj, name)
        if value < 0:
            raise HarnessError(f"--{name.replace('_', '-')} must be non-negative")


def make_timestamped_dir(output_root: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_rocprof_counters(
    *,
    rocprof: Path,
    executable: Path,
    run_dir: Path,
    app_stdin: str,
    kernel_regex: str | None,
    env: dict[str, str],
    verbose: bool,
    pmc_counters: list[str] | None = None,
    error_context: str = "rocprofv3 counter collection",
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(rocprof),
        "--output-format",
        "csv",
        "--output-directory",
        str(run_dir),
    ]
    if pmc_counters:
        cmd.append("--pmc")
        cmd.extend(pmc_counters)
    if kernel_regex:
        cmd.extend(["--kernel-include-regex", kernel_regex])
    cmd.extend(["--", str(executable)])

    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)

    result = subprocess.run(
        cmd,
        input=app_stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if verbose and result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if verbose and result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.returncode != 0:
        raise HarnessError(
            f"{error_context} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def run_application(
    *,
    executable: Path,
    app_stdin: str,
    env: dict[str, str],
    verbose: bool,
    error_context: str,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(executable)]
    if verbose:
        print("+ " + " ".join(cmd), file=sys.stderr)

    result = subprocess.run(
        cmd,
        input=app_stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if verbose and result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if verbose and result.stdout:
        print(result.stdout, file=sys.stderr, end="")
    if result.returncode != 0:
        raise HarnessError(
            f"{error_context} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def query_wavefront_threads(executable: Path, env: dict[str, str]) -> int:
    result = subprocess.run(
        [str(executable), "--print-wavefront-size"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise HarnessError(
            f"failed to query wavefront size from {executable} "
            f"with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    output = result.stdout.strip()
    try:
        wavefront_threads = int(output)
    except ValueError as exc:
        raise HarnessError(
            f"invalid wavefront size from {executable}: {output!r}\n"
            f"stderr:\n{result.stderr}"
        ) from exc

    if wavefront_threads <= 0:
        raise HarnessError(
            f"invalid wavefront size from {executable}: {wavefront_threads}"
        )
    return wavefront_threads


def find_counter_collection_csvs(run_dir: Path) -> list[Path]:
    matches = sorted(run_dir.rglob("*_counter_collection.csv"))
    if not matches:
        raise HarnessError(
            f"no rocprof counter collection CSV files produced under {run_dir}"
        )
    return matches


def parse_float(value: str, *, field: str, csv_path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise HarnessError(f"invalid float in {csv_path}: {field}={value!r}") from exc


def read_counter_value_sum(run_dir: Path, counter_name: str) -> float:
    values: list[float] = []
    for csv_path in find_counter_collection_csvs(run_dir):
        with csv_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            missing_columns = COUNTER_COLLECTION_REQUIRED_COLUMNS - set(
                reader.fieldnames or []
            )
            if missing_columns:
                raise HarnessError(
                    f"{csv_path} is missing required columns: "
                    f"{sorted(missing_columns)}"
                )
            for row in reader:
                if row["Counter_Name"] != counter_name:
                    continue
                values.append(
                    parse_float(
                        row["Counter_Value"],
                        field="Counter_Value",
                        csv_path=csv_path,
                    )
                )

    if not values:
        raise HarnessError(
            f"no {counter_name} rows found in counter CSV files under {run_dir}"
        )
    return sum(values)


def read_optional_lds_bank_conflict_count(
    run_dir: Path,
    counter_name: str,
) -> tuple[float | None, str | None]:
    try:
        return read_counter_value_sum(run_dir, counter_name), None
    except HarnessError as exc:
        return (
            None,
            "LDS bank-conflict PMC output unavailable; "
            "conflicts/workgroup will be blank for affected observations; "
            f"first error: {exc}",
        )


def normalize_lds_bank_conflict_count(
    raw_count: float,
    *,
    workgroup_count: int = PHASE_MASK_WORKGROUP_COUNT,
) -> tuple[float | None, str | None]:
    if workgroup_count <= 0:
        raise HarnessError(f"workgroup count must be positive: {workgroup_count}")
    value = raw_count / workgroup_count
    if not raw_count.is_integer():
        return (
            value,
            "raw LDS bank-conflict count is not integral; "
            "displayed conflicts/workgroup values may be nonsensical",
        )
    raw_int = int(raw_count)
    if raw_int % workgroup_count != 0:
        return (
            value,
            "raw LDS bank-conflict count is not divisible by launched "
            "workgroups; displayed conflicts/workgroup values may be nonsensical",
        )
    return value, None


def format_value(value: object, field: str | None = None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if field is not None:
            if field in {"lds_bank_conflict", "conflicts/workgroup"}:
                return f"{value:.0f}" if value.is_integer() else f"{value:.3f}"
            if field in {
                "min_thread_latency",
                "min thread latency",
                "max_thread_latency",
                "max thread latency",
            }:
                return f"{value:.0f}"
            if field in {"cycles_avg", "cycles/active thread", "avg thread latency"}:
                return f"{value:.3f}"
            if field in {"largest_gap", "second_largest_gap", "gap_margin"}:
                return f"{value:.1f}"
            if field.endswith("_ratio"):
                return f"{value:.3f}"
        return f"{value:g}"
    return str(value)


def print_table_header(header: list[TableColumn]) -> None:
    pieces = []
    for _, label, width in header:
        if width:
            pieces.append(f"{label:<{width}}")
        else:
            pieces.append(label)
    print(" ".join(pieces), flush=True)
    print(" ".join("-" * len(piece) for piece in pieces), flush=True)


def print_table_record(record: dict[str, object], columns: list[TableColumn]) -> None:
    pieces = []
    for column, _, width in columns:
        value = format_value(record[column], column)
        if width:
            pieces.append(f"{value[:width]:<{width}}")
        else:
            pieces.append(value)
    print(" ".join(pieces), flush=True)


def print_timer_table_header(label_header: str = "label") -> None:
    print_table_header(Profiler.timer_table_columns(label_header=label_header))


def print_timer_table_record(
    record: dict[str, object],
    *,
    label_header: str = "label",
) -> None:
    print_table_record(record, Profiler.timer_table_columns(label_header=label_header))


def summarize_raw_outputs(raw_outputs: list[str]) -> str:
    if raw_outputs and set(raw_outputs) == {"direct benchmark stdout"}:
        if len(raw_outputs) == 1:
            return raw_outputs[0]
        return f"{len(raw_outputs)} direct benchmark runs"
    if len(raw_outputs) == 1:
        return raw_outputs[0]
    parent = Path(raw_outputs[0]).parent
    return f"{len(raw_outputs)} dirs under {parent}"


def warning_dedup_key(warning: str) -> str:
    return warning.split("; first error:", 1)[0]


def unique_warnings(warnings: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for warning in warnings:
        key = warning_dedup_key(warning)
        if key in seen:
            continue
        seen.add(key)
        unique.append(warning)
    return unique


def parse_timing_output(stdout: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            continue
        try:
            values[key] = float(raw_value)
        except ValueError:
            continue
    if "cycles/active thread" in values:
        values["cycles_avg"] = values["cycles/active thread"]
    if "min thread latency" in values:
        values["min_thread_latency"] = values["min thread latency"]
    if "max thread latency" in values:
        values["max_thread_latency"] = values["max thread latency"]
    missing_fields = [field for field in TIMER_FIELDS if field not in values]
    if missing_fields:
        raise HarnessError(
            "timing benchmark output did not contain "
            f"{', '.join(missing_fields)}:\n{stdout}"
        )
    return values


def latency_metric_values(
    records_by_key: dict[ClassificationKey, dict[str, object]],
) -> tuple[str, dict[ClassificationKey, float]]:
    values: dict[ClassificationKey, float] = {}
    for key, record in records_by_key.items():
        value = record.get("cycles_avg")
        if not isinstance(value, float):
            raise HarnessError(
                f"record {key!r} has no cycles_avg value for latency classification"
            )
        values[key] = value
    return "avg thread latency", values


@dataclass(frozen=True)
class PhaseMaskObservation:
    timing: dict[str, float]
    lds_bank_conflict: float | None
    raw_output: str
    warnings: list[str]


class Profiler:
    def __init__(
        self,
        *,
        root: Path,
        rocprof: Path | None,
        executable: Path,
        output_root: Path,
        runs: int,
        kernel_regex: str | None,
        verbose: bool,
    ) -> None:
        if runs <= 0:
            raise HarnessError("--runs must be positive")
        validate_executable(executable, "benchmark executable")

        self.root = root
        self.rocprof = rocprof
        self.executable = executable
        self.output_root = output_root
        self.runs = runs
        self.kernel_regex = kernel_regex
        self.verbose = verbose
        self.env = rocprof_env(root)

    def __require_rocprof(self) -> Path:
        if self.rocprof is None:
            raise HarnessError("rocprofv3 is required for PMC counter collection")
        validate_executable(self.rocprof, "rocprofv3")
        return self.rocprof

    def query_thread_mask_wavefront_threads(self) -> int:
        wavefront_threads = query_wavefront_threads(self.executable, self.env)
        full_thread_mask(wavefront_threads)
        return wavefront_threads

    def __run_counters(
        self,
        *,
        app_stdin: str,
        run_dir: Path,
        error_context: str,
        pmc_counters: list[str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        rocprof = self.__require_rocprof()
        return run_rocprof_counters(
            rocprof=rocprof,
            executable=self.executable,
            run_dir=run_dir,
            app_stdin=app_stdin,
            kernel_regex=self.kernel_regex,
            env=self.env,
            verbose=self.verbose,
            pmc_counters=pmc_counters,
            error_context=error_context,
        )

    def __run_direct(
        self,
        *,
        app_stdin: str,
        error_context: str,
    ) -> subprocess.CompletedProcess[str]:
        return run_application(
            executable=self.executable,
            app_stdin=app_stdin,
            env=self.env,
            verbose=self.verbose,
            error_context=error_context,
        )

    def __run_phase_mask_timer_observation(
        self,
        *,
        run_prefix: str,
        run_index: int,
        app_stdin: str,
        error_context: str,
        pmc_counter: str | None,
    ) -> PhaseMaskObservation:
        warnings: list[str] = []
        lds_bank_conflict = None

        if pmc_counter is None:
            result = self.__run_direct(
                app_stdin=app_stdin,
                error_context=error_context,
            )
            raw_output = "direct benchmark stdout"
        else:
            run_dir = make_timestamped_dir(
                self.output_root,
                f"{run_prefix}_run_{run_index}",
            )
            raw_output = str(run_dir)
            try:
                result = self.__run_counters(
                    run_dir=run_dir,
                    app_stdin=app_stdin,
                    error_context=error_context,
                    pmc_counters=[pmc_counter],
                )
                (
                    raw_conflict_count,
                    counter_warning,
                ) = read_optional_lds_bank_conflict_count(
                    run_dir,
                    pmc_counter,
                )
                if raw_conflict_count is not None:
                    (
                        lds_bank_conflict,
                        normalize_warning,
                    ) = normalize_lds_bank_conflict_count(raw_conflict_count)
                    if normalize_warning is not None:
                        warnings.append(normalize_warning)
                if counter_warning is not None:
                    warnings.append(counter_warning)
            except HarnessError as exc:
                warnings.append(
                    "LDS bank-conflict PMC collection failed; "
                    "conflicts/workgroup will be blank for affected observations; "
                    f"first error: {exc}"
                )
                result = self.__run_direct(
                    app_stdin=app_stdin,
                    error_context=error_context,
                )
                raw_output = "direct benchmark stdout"

        return PhaseMaskObservation(
            timing=parse_timing_output(result.stdout),
            lds_bank_conflict=lds_bank_conflict,
            raw_output=raw_output,
            warnings=warnings,
        )

    @staticmethod
    def __average_phase_mask_observations(
        *,
        label: object,
        runs: int,
        observations: list[PhaseMaskObservation],
    ) -> tuple[dict[str, object], list[str]]:
        warnings: list[str] = []
        raw_outputs: list[str] = []
        lds_bank_conflict_counts: list[float] = []
        timer_records: list[dict[str, float]] = []

        for observation in observations:
            warnings.extend(observation.warnings)
            raw_outputs.append(observation.raw_output)
            timer_records.append(observation.timing)
            if observation.lds_bank_conflict is not None:
                lds_bank_conflict_counts.append(observation.lds_bank_conflict)

        lds_bank_conflict = None
        if lds_bank_conflict_counts:
            lds_bank_conflict = sum(lds_bank_conflict_counts) / len(
                lds_bank_conflict_counts
            )

        average_record: dict[str, object] = {
            "label": label,
            "runs": runs,
            "lds_bank_conflict": lds_bank_conflict,
            "raw_output": summarize_raw_outputs(raw_outputs),
        }
        for field in TIMER_FIELDS:
            values = [record[field] for record in timer_records if field in record]
            if not values:
                average_record[field] = None
            elif field == "min_thread_latency":
                average_record[field] = float(min(values))
            elif field == "max_thread_latency":
                average_record[field] = float(max(values))
            else:
                average_record[field] = float(mean(values))
        return average_record, unique_warnings(warnings)

    def profile_phase_mask_timer_observation(
        self,
        *,
        label: object,
        run_prefix: str,
        access_width: int,
        thread_mask: int,
        bank_count: int,
        threads: int,
        error_context: str,
        pmc_counter: str | None,
    ) -> tuple[dict[str, object], list[str]]:
        app_stdin = phase_mask_stdin(
            access_width=access_width,
            thread_mask=thread_mask,
            bank_count=bank_count,
            threads=threads,
        )
        observations = [
            self.__run_phase_mask_timer_observation(
                run_prefix=run_prefix,
                run_index=run_index,
                app_stdin=app_stdin,
                error_context=error_context,
                pmc_counter=pmc_counter,
            )
            for run_index in range(self.runs)
        ]
        return self.__average_phase_mask_observations(
            label=label,
            runs=self.runs,
            observations=observations,
        )

    def __available_pmc_counter_names(self) -> set[str]:
        rocprof = self.__require_rocprof()
        cmd = [
            str(rocprof),
            "--list-avail",
            "--",
            str(self.executable),
            "--print-wavefront-size",
        ]
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            check=False,
        )
        if result.returncode != 0:
            raise HarnessError(
                "failed to list rocprof counters with exit code "
                f"{result.returncode}\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        return set(COUNTER_NAME_PATTERN.findall(result.stdout + "\n" + result.stderr))

    def __choose_lds_bank_conflict_counter(self) -> str:
        available_counters = self.__available_pmc_counter_names()
        for counter_name in RAW_LDS_BANK_CONFLICT_COUNTERS:
            if counter_name in available_counters:
                return counter_name

        raise HarnessError(
            "no supported raw LDS bank-conflict counter was advertised by "
            "rocprofv3 --list-avail. Tried: "
            f"{', '.join(RAW_LDS_BANK_CONFLICT_COUNTERS)}. "
            "The derived LDSBankConflict metric is intentionally excluded."
        )

    def choose_optional_lds_bank_conflict_counter(
        self,
    ) -> tuple[str | None, str | None]:
        try:
            return self.__choose_lds_bank_conflict_counter(), None
        except HarnessError as exc:
            return (
                None,
                "LDS bank-conflict PMC unavailable; "
                f"conflicts/workgroup will be blank: {exc}",
            )

    @staticmethod
    def timer_table_columns(label_header: str = "label") -> list[TableColumn]:
        return [
            ("label", label_header, 8),
            ("runs", "runs", 4),
            ("lds_bank_conflict", "conflicts/workgroup", 19),
            ("cycles_avg", "avg thread latency", 19),
            ("min_thread_latency", "min thread latency", 19),
            ("max_thread_latency", "max thread latency", 19),
        ]
