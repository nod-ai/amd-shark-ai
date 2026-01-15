import os
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
DEST_DIRS = [
    "/home/amily/amd-shark-ai/amdsharktuner/dispatch_tuner/tuning_database_tf",
    "/home/amily/amd-shark-ai/amdsharktuner/dispatch_tuner/tuning_database_vd",
]

THRESHOLD = 0.60  # e.g., 90% compilation rate
CSV_EXTENSIONS = (".csv",)
# =========================================================


def is_true(value):
    """
    Normalizes truthy values that may appear in CSVs.
    Accepts True, 'true', '1', 1, 'yes', etc.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def process_csv(csv_path, threshold):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV: {csv_path} ({e})")
        return

    required_cols = {"compile_status", "to_compile"}
    if not required_cols.issubset(df.columns):
        return  # silently skip files without required columns

    to_compile_mask = df["to_compile"].apply(is_true)
    compile_status_mask = df["compile_status"].apply(is_true)

    to_compile_count = to_compile_mask.sum()
    if to_compile_count == 0:
        return  # avoid division by zero

    compiled_count = (to_compile_mask & compile_status_mask).sum()
    compilation_rate = compiled_count / to_compile_count

    if compilation_rate < threshold:
        # print(f"Directory   : {os.path.dirname(csv_path)}")
        print(f"CSV File    : {Path(csv_path).stem}")
        print(f"Rate        : {compilation_rate:.2%}")
        print("-" * 60)
    return compilation_rate


def scan_directory(base_dir, threshold):
    print(f"Directory   : {base_dir}")
    rates = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(CSV_EXTENSIONS):
                csv_path = os.path.join(root, file)
                rate = process_csv(csv_path, threshold)
                if rate is not None:
                    rates.append(rate)

    # ============================
    # Build histogram buckets 0–10, 10–20, … 90–100
    # ============================
    bins = [i/10 for i in range(11)]  # 0.0,0.1,...,1.0
    bucket_counts = [0] * 10  # 10 buckets

    for r in rates:
        idx = min(int(r * 10), 9)
        bucket_counts[idx] += 1

    labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]

    # ============================
    # Plot and save
    # ============================
    plt.figure(figsize=(10, 5))
    x = range(10)
    plt.bar(x, bucket_counts)

    plt.xticks(x, labels, rotation=45)
    plt.ylabel("Number of CSV Files")
    plt.title(f"MI300X: Compilation Rate Distribution in {Path(base_dir).name}")

    out_path = os.path.join(base_dir, "mi300x_compilation_rate.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"[INFO] Saved histogram to: {out_path}")


def main():
    for dest_dir in DEST_DIRS:
        if not os.path.isdir(dest_dir):
            print(f"[WARN] Skipping invalid directory: {dest_dir}")
            continue
        scan_directory(dest_dir, THRESHOLD)


if __name__ == "__main__":
    main()
