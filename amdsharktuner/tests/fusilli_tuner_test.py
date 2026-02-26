# Copyright 2026 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import os
import subprocess
from pathlib import Path
from typing import Callable
from unittest.mock import patch


from fusilli_tuner.fusilli_tuner import (
    load_commands_from_file_or_args,
    build_compile_args,
    insert_placeholder_input_file,
    find_cached_artifacts,
    parse_args,
    run_fusilli_benchmark_driver,
    FusilliPathConfig,
    FusilliTuner,
)


@pytest.fixture
def tmp_file(tmp_path: Path) -> Callable[[str, str], Path]:
    """Factory fixture for creating temporary files with content."""
    counter = 0

    def _create(content: str, suffix: str = ".txt") -> Path:
        nonlocal counter
        counter += 1
        temp_file = tmp_path / f"test_file_{counter}{suffix}"
        temp_file.write_text(content)
        return temp_file

    return _create


def test_load_commands_from_file_or_args(tmp_file: Callable[[str, str], Path]) -> None:
    """Test load_commands_from_file_or_args with files, args, and error cases."""
    # Test loading from file with trailing newline.
    content = """# Fusilli example commands
conv -F 1 --bf16 -n 1 -c 64 -H 28 -W 28 -k 128
matmul -M 1024 -N 1024 -K 1024 --a_type bf16
"""
    file_path = tmp_file(content, ".txt")
    result = load_commands_from_file_or_args(str(file_path), [])

    assert len(result) == 2
    assert result[0][0] == "conv"
    assert result[1][0] == "matmul"

    # Test loading from file without trailing newline.
    content_no_newline = """# Fusilli example commands
conv -F 1 --bf16 -n 1 -c 64
matmul -M 1024 -N 1024 -K 1024"""
    file_path_no_newline = tmp_file(content_no_newline, ".txt")
    result_no_newline = load_commands_from_file_or_args(str(file_path_no_newline), [])

    assert len(result_no_newline) == 2
    assert result_no_newline[0][0] == "conv"
    assert result_no_newline[1][0] == "matmul"
    assert "-M" in result_no_newline[1]
    assert "1024" in result_no_newline[1]

    # Test loading from args.
    fusilli_args = ["conv", "-F", "1", "--bf16", "-n", "1"]
    result = load_commands_from_file_or_args(None, fusilli_args)

    assert len(result) == 1
    assert result[0] == ["conv", "-F", "1", "--bf16", "-n", "1"]

    # Test tab-separated args (for TSV copy-paste support).
    fusilli_args = ["conv\t-F\t1", "--bf16"]
    result = load_commands_from_file_or_args(None, fusilli_args)

    assert len(result) == 1
    assert result[0] == ["conv", "-F", "1", "--bf16"]

    # Test error when both file and args are specified.
    content = "conv -F 1"
    file_path = tmp_file(content, ".txt")

    with pytest.raises(
        ValueError, match="Cannot specify both --commands-file and --fusilli-args"
    ):
        load_commands_from_file_or_args(str(file_path), ["conv", "-F", "1"])


def test_parse_args_fusilli_args_splitting() -> None:
    """Test that parse_args correctly splits --fusilli-args with shlex."""
    argv = [
        "fusilli_tuner",
        "--fusilli-args=conv -F 1 --bf16 -n 1 -c 64",
        "--devices=hip://0",
    ]

    args, fusilli_op_args = parse_args(argv)

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
    # Success case: auto-detect .mlir and .txt files by extension.
    base_dir = tmp_path / "success_cache"
    fusilli_cache = base_dir / ".cache" / "fusilli"
    graph_dir = fusilli_cache / "graph_12345"
    graph_dir.mkdir(parents=True)
    mlir_file = graph_dir / "some_input.mlir"
    command_file = graph_dir / "some_command.txt"
    mlir_file.write_text("module { }")
    command_file.write_text("iree-compile input.mlir")

    source_mlir, compile_command = find_cached_artifacts(base_dir)
    assert source_mlir == mlir_file
    assert compile_command == command_file
    assert source_mlir.exists()
    assert compile_command.exists()

    # Error case: base directory doesn't exist.
    with pytest.raises(FileNotFoundError, match="Fusilli cache not found"):
        find_cached_artifacts(tmp_path / "nonexistent_base")

    # Error case: empty cache (no graph directories).
    empty_base = tmp_path / "empty_base"
    (empty_base / ".cache" / "fusilli").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Expected exactly one graph directory"):
        find_cached_artifacts(empty_base)

    # Error case: multiple graph directories.
    multi_base = tmp_path / "multi_base"
    (multi_base / ".cache" / "fusilli" / "graph_1").mkdir(parents=True)
    (multi_base / ".cache" / "fusilli" / "graph_2").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="Expected exactly one graph directory"):
        find_cached_artifacts(multi_base)

    # Error case: no MLIR files.
    mlir_missing_base = tmp_path / "mlir_missing_base"
    mlir_missing_graph = mlir_missing_base / ".cache" / "fusilli" / "graph_1"
    mlir_missing_graph.mkdir(parents=True)
    (mlir_missing_graph / "command.txt").write_text("cmd")
    with pytest.raises(FileNotFoundError, match="Expected exactly one .mlir file"):
        find_cached_artifacts(mlir_missing_base)

    # Error case: no txt files.
    txt_missing_base = tmp_path / "txt_missing_base"
    txt_missing_graph = txt_missing_base / ".cache" / "fusilli" / "graph_2"
    txt_missing_graph.mkdir(parents=True)
    (txt_missing_graph / "input.mlir").write_text("module { }")
    with pytest.raises(FileNotFoundError, match="Expected exactly one .txt file"):
        find_cached_artifacts(txt_missing_base)

    # Error case: multiple MLIR files.
    multi_mlir_base = tmp_path / "multi_mlir_base"
    multi_mlir_graph = multi_mlir_base / ".cache" / "fusilli" / "graph_3"
    multi_mlir_graph.mkdir(parents=True)
    (multi_mlir_graph / "input1.mlir").write_text("module { }")
    (multi_mlir_graph / "input2.mlir").write_text("module { }")
    (multi_mlir_graph / "command.txt").write_text("cmd")
    with pytest.raises(FileNotFoundError, match="Expected exactly one .mlir file"):
        find_cached_artifacts(multi_mlir_base)

    # Error case: multiple txt files.
    multi_txt_base = tmp_path / "multi_txt_base"
    multi_txt_graph = multi_txt_base / ".cache" / "fusilli" / "graph_4"
    multi_txt_graph.mkdir(parents=True)
    (multi_txt_graph / "input.mlir").write_text("module { }")
    (multi_txt_graph / "command1.txt").write_text("cmd1")
    (multi_txt_graph / "command2.txt").write_text("cmd2")
    with pytest.raises(FileNotFoundError, match="Expected exactly one .txt file"):
        find_cached_artifacts(multi_txt_base)


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


def test_run_fusilli_benchmark_driver(tmp_path: Path) -> None:
    """Test run_fusilli_benchmark_driver with success and failure cases."""
    # Test successful execution.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    mock_success = subprocess.CompletedProcess(
        args=["fusilli_driver"], returncode=0, stdout="Fusilli output", stderr=""
    )

    with patch(
        "fusilli_tuner.fusilli_tuner.subprocess.run", return_value=mock_success
    ) as mock_run:
        run_fusilli_benchmark_driver("fusilli_driver", ["conv", "-F", "1"], cache_dir)

        # Verify subprocess.run was called with correct arguments.
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == [
            "fusilli_driver",
            "--dump",
            "--iter",
            "1",
            "conv",
            "-F",
            "1",
        ]
        assert call_args[1]["env"]["FUSILLI_CACHE_DIR"] == str(cache_dir)
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is True

    # Test failed execution.
    mock_failure = subprocess.CompletedProcess(
        args=["fusilli_driver"],
        returncode=1,
        stdout="Some output",
        stderr="Error message",
    )

    with patch("fusilli_tuner.fusilli_tuner.subprocess.run", return_value=mock_failure):
        with pytest.raises(
            RuntimeError, match="Fusilli benchmark driver failed with code 1"
        ):
            run_fusilli_benchmark_driver("fusilli_driver", ["matmul"], cache_dir)


def test_fusilli_path_config(tmp_path: Path) -> None:
    """Test FusilliPathConfig directory naming and benchmark config creation."""
    config = FusilliPathConfig()

    # Test directory naming with timestamp.
    base_dir = config._name_base_dir()
    assert base_dir.name.startswith("fusilli_tuning_")
    assert "fusilli_tuning_" in str(base_dir)

    # Test creating benchmark-specific PathConfig.
    config.base_dir = tmp_path / "main_tuning"
    benchmark_config = config.create_benchmark_path_config("conv_benchmark")
    assert benchmark_config.base_dir == tmp_path / "main_tuning" / "conv_benchmark"


def test_fusilli_tuner_getters() -> None:
    """Test FusilliTuner getter methods."""
    tuner = FusilliTuner(None)  # type: ignore[arg-type]
    tuner.compile_flags = ["--flag1", "--flag2"]
    tuner.benchmark_flags = ["--bench1"]
    tuner.compile_timeout = 30.0
    tuner.benchmark_timeout = 60.0
    tuner.auto_benchmark_timeout = False

    assert tuner.get_iree_compile_flags() == ["--flag1", "--flag2"]
    assert tuner.get_iree_benchmark_module_flags() == ["--bench1"]
    assert tuner.get_iree_compile_timeout_s() == 30.0
    assert tuner.get_iree_benchmark_timeout_s() == 60.0
    assert tuner.is_auto_iree_benchmark_timeout() is False
    assert tuner.should_prune_slower_candidates() is True
