import subprocess
from pathlib import Path
import os
import random
import time
from datetime import datetime
import logging
import shutil
import sys

random.seed(42) 
redo_list = [

]
ok_list = [

]
failed_list = [

]

DEVICE="hip://0,hip://1,hip://2,hip://3,hip://4,hip://5,hip://6,hip://7"
TUNING_TASKS=["llvmgpu_vector_distribute", "llvmgpu_tile_and_fuse"]
NUM_CAN=10000
TIMING_METHOD="rocprof"
SORT_METHOD="heuristic"
REP=5


def setup_logging() -> logging.Logger:
    base_dir = Path(os.path.abspath(__file__)).parent
    log_file_name = "tuning.log"
    run_log_path = base_dir / log_file_name

    # Create file handler for logging to a file.
    # file_handler = logging.FileHandler(run_log_path, mode="w")
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.DEBUG)

    # Create stream handler for logging to the console (only warnings and higher).
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that dynamically adds [levelname] for ERROR and WARNING.
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                return f"{record.message}"
            else:
                return f"[{record.levelname}] {record.message}"

    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_formatter = CustomFormatter()

    # Set formatters to handlers.
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Configure the root logger.
    logging.basicConfig(
        level=logging.DEBUG,  # Set the root logger to the lowest level.
        handlers=[file_handler, console_handler],
    )

    return logging.getLogger()


def append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python dispatches_tuner.py <arch>\nExample: python dispatches_tuner.py gfx942")
    arch = sys.argv[1]
    
    logger = setup_logging()
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    success_list_path = base_path / "success_list.log"
    failed_list_path = base_path / "failed_list.log"
    append_line(success_list_path, f"{arch}")
    append_line(failed_list_path, f"{arch}")
    
    logger.debug(f"Arch: {arch}")
    mlir_benchmark_folder_path = (base_path / "dump").expanduser().resolve()
    logger.debug(f"In MLIR_benchmark folder {mlir_benchmark_folder_path}: ")
    mlir_benchmark_files = sorted(mlir_benchmark_folder_path.glob("*.bench"))
    for f in mlir_benchmark_files:
        logger.debug(f"{f.stem}")

    logger.info(f"Found {len(mlir_benchmark_files)} benchmark file(s)")

    failed_files = []
    ok = fail = 0

    csv_dir_tf = base_path / "tuning_database_tf"
    csv_dir_tf.mkdir(exist_ok=True)
    
    csv_dir_vd = base_path / "tuning_database_vd"
    csv_dir_vd.mkdir(exist_ok=True)

    # --- timing + logging setup ---
    start_dt = datetime.now()
    start_perf = time.perf_counter()
    logger.debug(f"Tuning started at {start_dt.isoformat(timespec='seconds')}")
    var_list = [DEVICE, TUNING_TASKS, NUM_CAN, TIMING_METHOD, SORT_METHOD, REP]
    logger.info(f"Tuning Vars: {var_list}")

    for i, bench in enumerate(mlir_benchmark_files, start=1):
        mlir_filename = bench.stem
        logger.info(f"Checking file {i} / {len(mlir_benchmark_files)}")

        # Check list
        if mlir_filename in ok_list:
            logger.debug(f"Skipping file {mlir_filename} in OK list")
            continue
        if mlir_filename in failed_list:
            logger.debug(f"Skipping file {mlir_filename} in failed list")
            continue

        # Check file
        if not bench.exists():
            logger.warning(f"Can't find {bench}, skipping")
            fail += 1
            failed_files.append(bench.name)
            continue

        # logger.info("=" * 80)
        # logger.info("=" * 80)
        tuning_tasks = TUNING_TASKS
        for j, codegen_pipeline in enumerate(tuning_tasks, start=1):
            logger.info(f"Tuning {i} ({j}/{len(tuning_tasks)}) / {len(mlir_benchmark_files)}: {bench.name} - {codegen_pipeline}")
            file_start = time.perf_counter()
            logger.debug(f"File {bench} started at {start_dt.isoformat(timespec='seconds')}")
            cmd = [
                "python3", "-m", "dispatch_tuner",
                str(bench),
                str(bench),
                "--compile-flags-file=dispatch_tuner/compile_flags.txt",
                f"--devices={DEVICE}",
                F"--num-candidates={NUM_CAN}",
                f"--codegen-pipeline={codegen_pipeline}",
                f"--benchmark-timing-method={TIMING_METHOD}",
                f"--candidate-order={SORT_METHOD}",
            ]
            rc = subprocess.call(
                cmd,
                cwd=Path("~/amd-shark-ai/amdsharktuner").expanduser(),
                stdout=subprocess.DEVNULL
            )
        
            # End timer
            finished_at = datetime.now()
            elapsed = time.perf_counter() - file_start
        
            # Handle result
            if rc == 0:
                ok += 1
                elapsed_min = elapsed / 60.0
                logger.info(
                    f"{finished_at.isoformat(timespec='seconds')} - "
                    f"{bench.name}: completed in {elapsed_min:.2f} min"
                )
                append_line(success_list_path, f"{bench.name} - {codegen_pipeline} - {elapsed_min:.2f} mins")
                if elapsed < 60:
                    time.sleep(60 - elapsed) # Make sure next tuning folder is a new folder

                # Move CSV to tuning database
                folders = sorted(
                    base_path.parent.glob("tuning_2026_*"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                latest_folder = folders[0]
                csv_files = list(latest_folder.glob("*.csv"))
                if len(csv_files) > 1:
                    logger.warning(f"{latest_folder} has multiple csv")
                for csv_file in csv_files:
                    if codegen_pipeline == "llvmgpu_vector_distribute":
                        dst_dir = csv_dir_vd
                    elif codegen_pipeline == "llvmgpu_tile_and_fuse":
                        dst_dir = csv_dir_tf
                    try:
                        shutil.copy2(csv_file, dst_dir)
                        logging.debug(f"Copied {csv_file} -> {dst_dir}")
                    except:
                        logging.error(f"Fail to copy {csv_file} -> {dst_dir}")
            else:
                fail += 1
                failed_files.append(f"{bench.name} - {codegen_pipeline}")
                append_line(success_list_path, f"{bench.name} - {codegen_pipeline}")
                logger.warning(f"{finished_at.isoformat(timespec='seconds')} - {bench.name}: 'FAIL({rc})' in {elapsed:.2f}")

    # --- summary logging ---
    if failed_files:
        logger.warning(f"Failed bench files {len(failed_files)}):")
        for name in failed_files:
            logger.warning(f"- {name}")

    end_perf = time.perf_counter()
    total_elapsed = end_perf - start_perf

    logger.info("-" * 80)
    logger.info(f"SUMMARY: Success: {ok} | Fail: {fail}")
    logger.info(f"SUMMARY: Total elapsed: {total_elapsed:.2f}s")
    logger.info("-" * 80)


if __name__ == "__main__":
    main()