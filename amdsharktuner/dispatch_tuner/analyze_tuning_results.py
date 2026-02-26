#!/usr/bin/env python3
"""Analyze tuning_check_results.json: load MLIR files, extract seeds, dump report.

For each entry:
  - Better=True  → iterate tuned_spec_paths in order, try seed analysis for each.
                    Stop after collecting 3 specs with non-empty seeds, or all tried.
                    Merge seeds across the collected lists (deduplicated).
  - Better=False → analyze benchmark_file_path (single file).
  - Better=null  → skip.

Outputs a JSON report with all original data plus extracted seeds.

Usage:
  python analyze_tuning_results.py tuning_check_results.json
  python analyze_tuning_results.py tuning_check_results.json -o results.json
"""

import argparse
import json
import os

from tiling_to_seeds import analyze_mlir_file

DEFAULT_MAX_SEED_HITS = 3


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _merge_seed_lists(seed_lists: list[list[dict]]) -> list[dict]:
    """Merge multiple seed lists, removing duplicates."""
    seen = set()
    merged = []
    for seeds in seed_lists:
        for s in seeds:
            key = (
                s["best_subgroup_count_per_workgroup"],
                s["best_mn_tile_count_per_subgroup"],
                s["best_k_tile_count_per_subgroup"],
                s["best_k_element_count_per_subgroup"],
            )
            if key not in seen:
                seen.add(key)
                merged.append(s)
    return merged


def process_tuned_specs(specs: list[dict], max_hits: int = 0) -> dict:
    """Try specs in order, collect up to max_hits with non-empty seeds.

    If max_hits <= 0, try all specs (no limit).

    Returns dict with:
      hit_analyses: list of analyses that produced seeds
      merged_seeds: deduplicated union of all found seeds
      tried, hits, no_seeds, errors: counts
    """
    hit_analyses = []
    tried = 0
    hits = 0
    no_seeds = 0
    errors = 0

    for spec_entry in specs:
        if max_hits > 0 and hits >= max_hits:
            break

        spec_path = spec_entry.get("spec_path", "")
        if not spec_path or not os.path.isfile(spec_path):
            errors += 1
            continue

        tried += 1
        try:
            analysis = analyze_mlir_file(spec_path)
        except Exception as e:
            errors += 1
            continue

        if analysis["valid_seed_count"] > 0:
            hits += 1
            hit_analyses.append({
                "candidate_id": spec_entry.get("candidate_id"),
                "pct_of_baseline": spec_entry.get("pct_of_baseline"),
                "spec_path": spec_path,
                "analysis": analysis,
            })
        else:
            no_seeds += 1

    seed_lists = [h["analysis"]["valid_seeds"] for h in hit_analyses]
    merged = _merge_seed_lists(seed_lists)

    return {
        "hit_analyses": hit_analyses,
        "merged_seeds": merged,
        "merged_seed_count": len(merged),
        "tried": tried,
        "hits": hits,
        "no_seeds": no_seeds,
        "errors": errors,
    }


def process_benchmark(benchmark_path: str) -> dict:
    """Analyze a single benchmark file."""
    if not benchmark_path or not os.path.isfile(benchmark_path):
        return {"error": f"File not found: {benchmark_path}"}

    try:
        analysis = analyze_mlir_file(benchmark_path)
    except Exception as e:
        return {"error": str(e)}

    return {
        "analysis": analysis,
        "merged_seeds": analysis["valid_seeds"],
        "merged_seed_count": analysis["valid_seed_count"],
    }


def process_entry(entry: dict, max_hits: int = 0) -> dict:
    """Process a single JSON entry."""
    result = {**entry, "source": None, "result": None, "skip_reason": None}

    better = entry.get("Better")

    if better is True:
        specs = entry.get("tuned_spec_paths") or []
        if specs:
            spec_result = process_tuned_specs(specs, max_hits=max_hits)
            if spec_result["merged_seed_count"] > 0:
                result["source"] = "tuned_specs"
                result["result"] = spec_result
                return result

        bm_path = entry.get("benchmark_file_path", "")
        if bm_path:
            result["source"] = "tuned_specs_fallback_benchmark"
            result["result"] = process_benchmark(bm_path)
        else:
            result["skip_reason"] = ("Better=True but no seeds from tuned specs "
                                     "and no benchmark_file_path")

    elif better is False:
        bm_path = entry.get("benchmark_file_path", "")
        if not bm_path:
            result["skip_reason"] = "Better=False but benchmark_file_path is empty"
            return result
        result["source"] = "benchmark"
        result["result"] = process_benchmark(bm_path)

    else:
        result["skip_reason"] = "Better is null (no tuning data)"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tuning results JSON and extract seeds from MLIR files."
    )
    parser.add_argument("input_file", help="Path to tuning_check_results.json")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON file path (default: <input_basename>_with_seeds.json)",
    )
    parser.add_argument(
        "--max-hits", type=int, default=DEFAULT_MAX_SEED_HITS,
        help="Max spec hits to collect per entry (0 = try all specs). "
             f"Default: {DEFAULT_MAX_SEED_HITS}",
    )
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.input_file)[0]
        args.output = f"{base}_with_seeds.json"

    entries = load_json(args.input_file)
    print(f"Loaded {len(entries)} entries from {args.input_file}")

    results = []
    skipped = 0
    analyzed = 0

    for i, entry in enumerate(entries):
        name = entry.get("filename", f"row_{i}")
        short = name[:60] + "..." if len(name) > 60 else name

        result = process_entry(entry, max_hits=args.max_hits)
        results.append(result)

        if result["skip_reason"]:
            skipped += 1
            print(f"  [{i+1}/{len(entries)}] SKIP  {short}")
            print(f"           {result['skip_reason']}")
            continue

        analyzed += 1
        r = result["result"]

        if result["source"] == "tuned_specs":
            print(f"  [{i+1}/{len(entries)}] OK    {short}")
            print(f"           tuned_specs: tried={r['tried']}, "
                  f"hits={r['hits']}, no_seeds={r['no_seeds']}, "
                  f"errors={r['errors']}")
            print(f"           merged_seeds={r['merged_seed_count']}")
            for h in r["hit_analyses"]:
                a = h["analysis"]
                cfg = a["lowering_config"]
                print(f"             hit cand={h['candidate_id']} "
                      f"pct={h['pct_of_baseline']} "
                      f"mma={cfg['mma_kind']} "
                      f"seeds={a['valid_seed_count']} "
                      f"sg={a['unique_sg_values']} mn={a['unique_mn_values']}")
        elif result["source"] == "tuned_specs_fallback_benchmark":
            if "error" in r:
                print(f"  [{i+1}/{len(entries)}] ERR   {short}")
                print(f"           tuned_specs: 0 seeds -> fallback benchmark")
                print(f"           {r['error']}")
            else:
                a = r["analysis"]
                cfg = a["lowering_config"]
                print(f"  [{i+1}/{len(entries)}] OK    {short}")
                print(f"           tuned_specs: 0 seeds -> fallback benchmark")
                print(f"           benchmark: mma={cfg['mma_kind']} "
                      f"seeds={a['valid_seed_count']} "
                      f"sg={a['unique_sg_values']} mn={a['unique_mn_values']}")
        else:
            if "error" in r:
                print(f"  [{i+1}/{len(entries)}] ERR   {short}")
                print(f"           {r['error']}")
            else:
                a = r["analysis"]
                cfg = a["lowering_config"]
                print(f"  [{i+1}/{len(entries)}] OK    {short}")
                print(f"           benchmark: mma={cfg['mma_kind']} "
                      f"seeds={a['valid_seed_count']} "
                      f"sg={a['unique_sg_values']} mn={a['unique_mn_values']}")

    # Build summary
    all_mn_sets = []
    for r in results:
        res = r.get("result")
        if not res or "error" in res:
            continue
        merged = res.get("merged_seeds", [])
        if not merged:
            continue
        mn_set = set()
        for s in merged:
            mn_set.add((
                s["best_subgroup_count_per_workgroup"],
                s["best_mn_tile_count_per_subgroup"],
            ))
        all_mn_sets.append(mn_set)

    common_mn = []
    if all_mn_sets:
        common = set(all_mn_sets[0])
        for s in all_mn_sets[1:]:
            common &= s
        common_mn = [{"best_subgroup": sg, "best_mn_tile": mn}
                     for sg, mn in sorted(common)]

    report = {
        "source_file": os.path.abspath(args.input_file),
        "total_entries": len(entries),
        "analyzed": analyzed,
        "skipped": skipped,
        "max_seed_hits_per_entry": args.max_hits,
        "common_mn_seeds": common_mn,
        "entries": results,
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Summary: {analyzed} analyzed, {skipped} skipped")
    if common_mn:
        print(f"Common MN seeds (across entries with seeds): {common_mn}")
    else:
        print("No common MN seeds across all analyzed entries with seeds.")
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
