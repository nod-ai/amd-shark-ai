#!/usr/bin/env python3
"""
MLIR File Processor
Processes MLIR files in a given folder with iree-compile and extracts benchmark files.
"""

import os
import subprocess
from pathlib import Path


def run_iree_compile(mlir_file, dump_dir, arch):
    """Run iree-compile command on the MLIR file."""
    cmd = [
        "iree-compile",
        str(mlir_file),
        "--iree-hal-target-device=hip",
        f"--iree-hip-target={arch}",
        f"--iree-hal-dump-executable-files-to={dump_dir}",
        "--iree-config-add-tuner-attributes",
        "-o", "/dev/null"
    ]
    
    print(f"Processing {mlir_file.name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {mlir_file.name}: {e}")
        print(f"stderr: {e.stderr}")
        return False


def copy_benchmark_file(input_name, src_dir, dst_dir):
    """Copy the benchmark file with renamed output."""
    source_file = src_dir / "module_main_dispatch_0_rocm_hsaco_fb_benchmark.mlir"
    dest_file = dst_dir / f"{input_name}_benchmark.mlir"
    
    if source_file.exists():
        try:
            subprocess.run(["cp", str(source_file), str(dest_file)], check=True)
            print(f"  → Created {dest_file.name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error copying benchmark file for {input_name}: {e}")
            return False
    else:
        print(f"Warning: Benchmark file not found for {input_name}")
        return False

import sys

def main():
    # parser = argparse.ArgumentParser(description="Process MLIR files with iree-compile")
    # parser.add_argument("input_dir", help="Path to folder containing MLIR files")
    # parser.add_argument("output_folder", help="Path to folder containing MLIR files")
    # args = parser.parse_args()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python compile_dump_exe.py <arch>\nExample: python compile_dump_exe.py gfx942")
    arch = sys.argv[1]

    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    input_dir = base_path / "problem_mlir_dump"
    
    print(f"Processing MLIR files in: {input_dir}")
    
    # Find all MLIR files
    mlir_files = list(input_dir.glob("*.mlir"))
    print(f"Found {len(mlir_files)} MLIR files to process")
    
    # Ensure output directory exists
    output_dir = Path(base_path) / "bench_dump"
    dump_dir = output_dir / "tmp"
    output_dir.mkdir(exist_ok=True)
    dump_dir.mkdir(exist_ok=True)
    
    # Process each file
    success_count = 0
    for i, mlir_file in enumerate(mlir_files, start=1):
        input_name = mlir_file.stem  # filename without extension
        
        out_file = f"{input_name}_benchmark.mlir"
        dest_file = output_dir / out_file
        if dest_file.exists():
            print(f"{out_file} already exists, skipping...")
            success_count += 1
            print(f"[{i} / {len(mlir_files)}]")
            continue
        # Run iree-compile
        if run_iree_compile(mlir_file, dump_dir, arch):
            # Copy benchmark file
            if copy_benchmark_file(input_name, dump_dir, output_dir):
                success_count += 1
        else:
            print(f"Failed on {input_name}")
        print(f"[{i} / {len(mlir_files)}]")
    
    print(f"Successfully processed {success_count}/{len(mlir_files)} files in {output_dir}")
    return success_count == len(mlir_files)


if __name__ == "__main__":
    main()