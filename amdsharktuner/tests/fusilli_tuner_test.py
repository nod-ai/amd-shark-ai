# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
from pathlib import Path
from typing import Callable


from fusilli_tuner.fusilli_tuner import (
    load_commands_from_file_or_args,
    build_compile_args,
    insert_placeholder_input_file,
    find_cached_artifacts,
    parse_args,
)


@pytest.fixture
def tmp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary files.

    Returns a callable that takes (content, suffix) and returns a Path to a
    temporary file with that content and suffix.
    """
    counter = 0

    def _create(content: str, suffix: str = ".txt") -> Path:
        nonlocal counter
        counter += 1
        temp_file = tmp_path / f"test_file_{counter}{suffix}"
        temp_file.write_text(content)
        return temp_file

    return _create


def test_load_fusilli_commands_from_file(tmp_file: Callable[[str, str], Path]) -> None:
    """Test loading Fusilli-specific commands from file."""
    content = """# Fusilli example commands
conv -F 1 --bf16 -n 1 -c 64 -H 28 -W 28 -k 128
matmul -M 1024 -N 1024 -K 1024 --a_type bf16
"""
    file_path = tmp_file(content, ".txt")

    result = load_commands_from_file_or_args(str(file_path), [])

    assert len(result) == 2
    assert result[0][0] == "conv"
    assert result[1][0] == "matmul"


def test_load_fusilli_commands_from_args() -> None:
    """Test loading Fusilli commands from --fusilli-args."""
    # Test with simple args.
    fusilli_args = ["conv", "-F", "1", "--bf16", "-n", "1"]
    result = load_commands_from_file_or_args(None, fusilli_args)

    assert len(result) == 1
    assert result[0] == ["conv", "-F", "1", "--bf16", "-n", "1"]

    # Test tab-separated args (for TSV copy-paste support).
    fusilli_args = ["conv\t-F\t1", "--bf16"]
    result = load_commands_from_file_or_args(None, fusilli_args)

    assert len(result) == 1
    assert result[0] == ["conv", "-F", "1", "--bf16"]


def test_load_commands_error_both_file_and_args(
    tmp_file: Callable[[str, str], Path]
) -> None:
    """Test error when both --commands-file and --fusilli-args are specified."""
    content = "conv -F 1"
    file_path = tmp_file(content, ".txt")

    with pytest.raises(
        ValueError, match="Cannot specify both --commands-file and --fusilli-args"
    ):
        load_commands_from_file_or_args(str(file_path), ["conv", "-F", "1"])


def test_parse_args_fusilli_args_splitting(monkeypatch) -> None:
    """Test that parse_args correctly splits --fusilli-args with shlex."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "fusilli_tuner",
            "--fusilli-args=conv -F 1 --bf16 -n 1 -c 64",
            "--devices=hip://0",
        ],
    )

    args, fusilli_op_args = parse_args()

    # Verify fusilli args were properly split by shlex.
    assert fusilli_op_args == ["conv", "-F", "1", "--bf16", "-n", "1", "-c", "64"]

    # Verify other args were parsed.
    assert args.devices == ["hip://0"]


def test_insert_placeholder_input_file() -> None:
    """Test that insert_placeholder_input_file inserts 'fusilli.mlir' after program name."""
    # Test with minimal argv.
    argv = ["fusilli_tuner"]
    result = insert_placeholder_input_file(argv)
    assert result == ["fusilli_tuner", "fusilli.mlir"]

    # Test with argv containing additional arguments.
    argv = ["fusilli_tuner", "--devices=hip://0", "--num-candidates=10"]
    result = insert_placeholder_input_file(argv)
    assert result == [
        "fusilli_tuner",
        "fusilli.mlir",
        "--devices=hip://0",
        "--num-candidates=10",
    ]

    # Test that original argv is not modified.
    original_argv = ["fusilli_tuner", "--flag"]
    original_copy = original_argv.copy()
    insert_placeholder_input_file(original_argv)
    assert original_argv == original_copy


def test_find_cached_artifacts(tmp_path: Path) -> None:
    """Test find_cached_artifacts for success and error cases."""
    cache_dir = tmp_path / "success_cache"
    fusilli_cache = cache_dir / ".cache" / "fusilli"
    graph_dir = fusilli_cache / "graph_12345"
    graph_dir.mkdir(parents=True)
    mlir_file = graph_dir / "iree-compile-input.mlir"
    command_file = graph_dir / "iree-compile-command.txt"
    mlir_file.write_text("module { }")
    command_file.write_text("iree-compile input.mlir")

    source_mlir, compile_command = find_cached_artifacts(cache_dir)
    assert source_mlir == mlir_file
    assert compile_command == command_file
    assert source_mlir.exists()
    assert compile_command.exists()

    # Error case: cache directory doesn't exist.
    with pytest.raises(FileNotFoundError, match="Fusilli cache not found"):
        find_cached_artifacts(tmp_path / "nonexistent_cache")

    # Error case: empty cache (no graph directories).
    empty_cache = tmp_path / "empty_cache"
    (empty_cache / ".cache" / "fusilli").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No graph directories found"):
        find_cached_artifacts(empty_cache)

    # Error case: missing MLIR file.
    mlir_missing_cache = tmp_path / "mlir_missing_cache"
    mlir_missing_graph = mlir_missing_cache / ".cache" / "fusilli" / "graph_1"
    mlir_missing_graph.mkdir(parents=True)
    (mlir_missing_graph / "iree-compile-command.txt").write_text("cmd")
    with pytest.raises(FileNotFoundError, match="Source MLIR not found"):
        find_cached_artifacts(mlir_missing_cache)

    # Error case: missing compile command file.
    cmd_missing_cache = tmp_path / "cmd_missing_cache"
    cmd_missing_graph = cmd_missing_cache / ".cache" / "fusilli" / "graph_2"
    cmd_missing_graph.mkdir(parents=True)
    (cmd_missing_graph / "iree-compile-input.mlir").write_text("module { }")
    with pytest.raises(FileNotFoundError, match="Compile command not found"):
        find_cached_artifacts(cmd_missing_cache)


def test_build_compile_args() -> None:
    """Test that build_compile_args filters unwanted flags and adds tuner flags."""
    compile_command = (
        "iree-compile --iree-hal-target-backends=rocm "
        "--iree-scheduling-dump-statistics-format=json "
        "--iree-scheduling-dump-statistics-file=/tmp/stats.json "
        "/path/to/input.mlir -o /path/to/output.vmfb"
    )
    benchmarks_dir = Path("/tmp/benchmarks")

    result = build_compile_args(compile_command, benchmarks_dir)

    # Statistics flags should be filtered out (combined with "=" as Fusilli generates them).
    assert all("--iree-scheduling-dump-statistics-format" not in arg for arg in result)
    assert all("--iree-scheduling-dump-statistics-file" not in arg for arg in result)
    assert "/path/to/output.vmfb" not in result

    assert "--iree-hal-target-backends=rocm" in result
    assert "/path/to/input.mlir" in result

    assert "--iree-config-add-tuner-attributes" in result
    assert "--iree-hal-dump-executable-benchmarks-to" in result
    assert str(benchmarks_dir) in result
    assert result[-2] == "-o"
    assert result[-1] == os.devnull
