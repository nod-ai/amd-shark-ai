#!/usr/bin/env python3
"""Reverse-engineer GPUMMAHeuristicSeeds from observed lowering_config tiling."""

import math
from math import gcd
import sys

sys.path.insert(0, "/home/amily/amd-shark-ai/amdsharktuner/dispatch_tuner")
from seed_to_tiling import compute_schedule, compute_lowering_config, divide_ceil, product

INTRINSIC_M = [16]
INTRINSIC_N = [16]
INTRINSIC_K = [16]

ENTRIES = [
    ("bmm_1cafd6d", 96, 96, 96, 10,
     [1,32,32,0], [0,1,1,0], [0,0,0,2], True, None),
    ("mm_9941b00", 1280, 3840, 576, 1,
     [64,128,0], [2,4,0], [0,0,4], False, None),
    ("mm_c80ec1f", 1280, 576, 1152, 1,
     [64,128,0], [2,4,0], [0,0,8], False, [64,128,128]),
    ("mm_434de99", 7680, 576, 576, 1,
     [64,128,0], [2,4,0], [0,0,4], False, [64,128,64]),
    ("mm_072405f", 32768, 512, 128, 1,
     [64,128,0], [2,4,0], [0,0,8], False, None),
    ("mm_e7f9a2d", 20, 3840, 21760, 1,
     [32,64,0], [1,2,0], [0,0,8], False, [32,64,128]),
    ("mm_23fdea0", 2048, 2048, 1285, 1,
     [128,128,0], [4,4,0], [0,0,1], False, [128,128,16]),
    ("mm_a55123", 24576, 2048, 512, 1,
     [64,128,0], [2,4,0], [0,0,8], False, None),
    ("bmm_65b381", 384, 192, 384, 5,
     [1,64,128,0], [0,2,4,0], [0,0,0,8], True, [1,64,128,128]),
    ("mm_c1733d", 10, 576, 2304, 1,
     [16,64,0], [1,2,0], [0,0,8], False, [16,64,128]),
    ("bmm_70dea6", 192, 384, 384, 16,
     [1,64,128,0], [0,2,4,0], [0,0,0,8], True, None),
    ("mm_fc8839", 576, 576, 1280, 1,
     [64,64,0], [2,2,0], [0,0,8], False, None),
    ("fused_mm_ebd6e", 16, 1024, 512, 1,
     [16,64,0], [1,2,0], [0,0,8], False, None),
    ("bmm_b3ae81", 384, 384, 192, 16,
     [1,64,128,0], [0,2,4,0], [0,0,0,4], True, None),
    ("mm_ac5438", 24576, 2048, 576, 1,
     [64,128,0], [2,4,0], [0,0,4], False, None),
]


def pad_up(val, multiple):
    if multiple == 0:
        return val
    return ((val + multiple - 1) // multiple) * multiple


def get_effective_sizes(entry):
    name, M, N, K, batch, wg, sg, red, is_bmm, padding = entry
    if padding is None:
        return M, N, K
    if is_bmm:
        return pad_up(M, padding[1]), pad_up(N, padding[2]), pad_up(K, padding[3])
    return pad_up(M, padding[0]), pad_up(N, padding[1]), pad_up(K, padding[2])


def get_dims(is_bmm):
    if is_bmm:
        return [1], [2], [3], [0], 4
    return [0], [1], [2], [], 3


def try_seeds(entry, best_sg, best_mn, best_kt, best_ke):
    name, M_raw, N_raw, K_raw, batch, twg, tsg, tred, is_bmm, padding = entry
    M, N, K = get_effective_sizes(entry)
    md, nd, kd, bd, rank = get_dims(is_bmm)
    m_s, n_s, k_s = [M], [N], [K]
    r = compute_schedule(m_s, n_s, k_s, INTRINSIC_M, INTRINSIC_N, INTRINSIC_K,
                         best_sg, best_mn, best_kt, best_ke)
    wg, sg, red = compute_lowering_config(rank, md, nd, kd, bd,
                                          INTRINSIC_M, INTRINSIC_N, *r)
    return wg == twg and sg == tsg and red == tred


def find_valid_mn(entry):
    """Find MN seeds by checking only workgroup/subgroup on M/N/batch dims."""
    name, M_raw, N_raw, K_raw, batch, twg, tsg, tred, is_bmm, padding = entry
    M, N, K = get_effective_sizes(entry)
    md, nd, kd, bd, rank = get_dims(is_bmm)
    m_s, n_s, k_s = [M], [N], [K]
    check_dims = md + nd + bd

    valid = set()
    for best_sg in [1 << i for i in range(6)]:
        for best_mn in [1 << i for i in range(6)]:
            try:
                r = compute_schedule(m_s, n_s, k_s,
                                     INTRINSIC_M, INTRINSIC_N, INTRINSIC_K,
                                     best_sg, best_mn, 1, 0)
                wg, sg, red = compute_lowering_config(rank, md, nd, kd, bd,
                                                      INTRINSIC_M, INTRINSIC_N, *r)
                match = all(wg[d] == twg[d] and sg[d] == tsg[d] for d in check_dims)
                if match:
                    valid.add((best_sg, best_mn))
            except Exception:
                pass
    return valid


def find_valid_k(K_eff, target_k_tile):
    k_total = divide_ceil(K_eff, 16)
    valid = set()
    for best_kt in range(1, 65):
        for best_ke in [0] + list(range(1, 1025)):
            eff = best_kt if best_ke == 0 else divide_ceil(best_ke, 16)
            if gcd(k_total, eff) == target_k_tile:
                valid.add((best_kt, best_ke))
    return valid


print("=" * 90)
print("STEP 1: Valid (best_subgroup, best_mn_tile) per entry  [with padding where applicable]")
print("=" * 90)

all_mn = []
for entry in ENTRIES:
    mn = find_valid_mn(entry)
    all_mn.append(mn)
    M_eff, N_eff, K_eff = get_effective_sizes(entry)
    pad = f"  [padded M={M_eff} N={N_eff} K={K_eff}]" if entry[9] else ""
    print(f"  {entry[0]:20s}  M={entry[1]:6d} N={entry[2]:6d} K={entry[3]:6d}  => {sorted(mn)}{pad}")

mn_inter = set(all_mn[0])  # copy!
for s in all_mn[1:]:
    mn_inter &= s
print(f"\n  Full intersection: {sorted(mn_inter)}")

# Also find largest subset that shares a common MN seed
print("\n  Per-seed coverage:")
all_possible_mn = set()
for s in all_mn:
    all_possible_mn |= s
for seed in sorted(all_possible_mn):
    matching = [ENTRIES[i][0] for i in range(len(ENTRIES)) if seed in all_mn[i]]
    if len(matching) >= 10:
        print(f"    {seed}: {len(matching)}/{len(ENTRIES)} entries match")

print("\n" + "=" * 90)
print("STEP 2: Valid K seeds per entry")
print("=" * 90)

all_k = []
for entry in ENTRIES:
    _, _, K_eff = get_effective_sizes(entry)
    is_bmm = entry[8]
    k_dim = 3 if is_bmm else 2
    target_k = entry[7][k_dim]
    kv = find_valid_k(K_eff, target_k)
    all_k.append(kv)
    print(f"  {entry[0]:20s}  K_eff={K_eff:6d}  target_k_tile={target_k}")

k_inter = set(all_k[0])
for s in all_k[1:]:
    k_inter &= s
print(f"\n  K intersection: {len(k_inter)} pairs")

print("\n" + "=" * 90)
print("STEP 3: Full verification — find entries that match (4,8) and which don't")
print("=" * 90)

target_mn = (4, 8)
print(f"\nTesting MN seed {target_mn}:")
for i, entry in enumerate(ENTRIES):
    matches_mn = target_mn in all_mn[i]
    print(f"  {entry[0]:20s}  matches={matches_mn}  "
          f"all_valid_mn={sorted(all_mn[i])}")

print(f"\n\nEntries NOT matching (4,8):")
outliers = []
for i, entry in enumerate(ENTRIES):
    if target_mn not in all_mn[i]:
        outliers.append(entry[0])
        print(f"  {entry[0]:20s}  only valid: {sorted(all_mn[i])}")

print(f"\n{'='*90}")
print("STEP 4: Verify (4,8) for all except outlier, and check outlier separately")
print(f"{'='*90}")

# Check if all non-outlier entries work with (4,8,kt,ke) for some common K seeds
non_outlier_k = None
outlier_k = None
for i, entry in enumerate(ENTRIES):
    _, _, K_eff = get_effective_sizes(entry)
    is_bmm = entry[8]
    k_dim = 3 if is_bmm else 2
    target_k = entry[7][k_dim]
    kv = find_valid_k(K_eff, target_k)
    if entry[0] in outliers:
        if outlier_k is None:
            outlier_k = set(kv)
        else:
            outlier_k &= kv
    else:
        if non_outlier_k is None:
            non_outlier_k = set(kv)
        else:
            non_outlier_k &= kv

# Now do full verification for non-outliers with (4, 8, kt, ke)
print(f"\nFull verification for {len(ENTRIES)-len(outliers)} non-outlier entries with MN=(4,8):")
valid_48 = []
for kt, ke in sorted(non_outlier_k or set()):
    all_ok = True
    for i, entry in enumerate(ENTRIES):
        if entry[0] in outliers:
            continue
        try:
            if not try_seeds(entry, 4, 8, kt, ke):
                all_ok = False
                break
        except:
            all_ok = False
            break
    if all_ok:
        valid_48.append((4, 8, kt, ke))

print(f"  Found {len(valid_48)} fully valid 4-tuples for non-outlier entries")
if valid_48:
    print(f"  Sample: {valid_48[:5]}")
    kt_set = set(s[2] for s in valid_48)
    ke_set = set(s[3] for s in valid_48)
    print(f"  Unique kt: {sorted(kt_set)[:10]}")
    print(f"  ke range: [{min(ke_set)}, {max(ke_set)}], count={len(ke_set)}")

# Also check outlier with its valid MN
if outliers:
    print(f"\nOutlier analysis ({outliers}):")
    for oi, entry in enumerate(ENTRIES):
        if entry[0] not in outliers:
            continue
        valid_seeds_outlier = sorted(all_mn[oi])
        print(f"  {entry[0]}: valid MN = {valid_seeds_outlier}")
        for omn in valid_seeds_outlier:
            count = 0
            for kt, ke in sorted(outlier_k or set())[:1]:
                try:
                    if try_seeds(entry, omn[0], omn[1], kt, ke):
                        count += 1
                except:
                    pass
            if count:
                print(f"    MN={omn} works with K seeds")

print(f"\n{'='*90}")
print("CONCLUSION")
print(f"{'='*90}")
if not outliers:
    print("All entries share the same seeds!")
else:
    print(f"14/15 entries are consistent with seeds: best_subgroup=4, best_mn_tile=8")
    print(f"Outlier: {outliers}")
    for entry in ENTRIES:
        if entry[0] in outliers:
            oi = ENTRIES.index(entry)
            print(f"  {entry[0]} (M={entry[1]}, N={entry[2]}, K={entry[3]}) "
                  f"requires MN seeds: {sorted(all_mn[oi])}")
    print(f"\nThe extrapolated seeds are NOT all the same.")
    print(f"  Majority seed: best_subgroup_count_per_workgroup=4, best_mn_tile_count_per_subgroup=8")
    print(f"  K seeds have inherent ambiguity (many (kt, ke) pairs produce the same tiling)")
