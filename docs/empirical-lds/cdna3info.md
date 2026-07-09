# CDNA3 MI300X LDS Measurements

Measurements were collected on MI300X (`gfx942`), which has a wavefront size of
64 threads. The LDS bank count was measured to be 32. With 4-byte banks, the
bank mapping repeats every 128 bytes, so an access stride of 128 bytes causes
the maximum possible number of bank conflicts.

Each table starts at the natural byte stride for that access width: 4 bytes for
`ds_read_b32`, 8 bytes for `ds_read_b64`, and 16 bytes for `ds_read_b128`. The
values were collected from `profile_bank_count.py` with 3 runs per stride
using the shared 64-read `lds_phase_mask` timing block. Bank-conflict counts
are collected with rocprof, normalized by the 4096 launched workgroups, and
cover the full 64-read block. Latencies are average active-thread timer values
from `clock64()`, reported in cycles, and were collected using the macro
payload with 64 `ds_read` instructions.

## ds_read_b32 Latencies

| stride (bytes) | bank conflicts per block | avg thread latency (cycles) |
| -------------- | ------------------------ | --------------------------- |
| 4              | 0                        | 500.756                     |
| 8              | 128                      | 570.156                     |
| 16             | 384                      | 956.776                     |
| 32             | 896                      | 1833.150                    |
| 64             | 1920                     | 3511.675                    |
| 128            | 3968                     | 6810.440                    |
| 256            | 3968                     | 6781.011                    |
| 512            | 3968                     | 6798.651                    |

When performing a `ds_read_b32`, lanes access LDS in 2 phases: T0-T31, then
T32-T63. The fully conflicted case has 31 conflicts in each phase, or 62
conflicts per `ds_read`. Across the 64-read timing block, that saturates at
3968 conflicts per block.

## ds_read_b64 Latencies

| stride (bytes) | bank conflicts per block | avg thread latency (cycles) |
| -------------- | ------------------------ | --------------------------- |
| 8              | 0                        | 852.258                     |
| 16             | 256                      | 991.498                     |
| 32             | 768                      | 1855.603                    |
| 64             | 1792                     | 3625.288                    |
| 128            | 3840                     | 7157.675                    |
| 256            | 3840                     | 7177.591                    |
| 512            | 3840                     | 7095.679                    |

For `ds_read_b64`, the access happens in four phases of 16 lanes each:
T0-T15, T16-T31, T32-T47, then T48-T63. The fully conflicted case has 15
conflicts per phase, or 60 conflicts per `ds_read`. Across the 64-read timing
block, that saturates at 3840 conflicts per block.

## ds_read_b128 Latencies

| stride (bytes) | bank conflicts per block | avg thread latency (cycles) |
| -------------- | ------------------------ | --------------------------- |
| 16             | 0                        | 1384.460                    |
| 32             | 512                      | 1941.316                    |
| 64             | 1536                     | 3803.940                    |
| 128            | 3584                     | 7614.521                    |
| 256            | 3584                     | 7545.271                    |
| 512            | 3584                     | 7573.651                    |

For `ds_read_b128`, the access happens in eight phases of 8 lanes each.

1. T0-T3 and T20-T23
2. T32-T35 and T52-T55
3. T4-T7 and T16-T19
4. T36-T39 and T48-T51
5. T8-T11 and T28-T31
6. T40-T43 and T60-T63
7. T12-T15 and T24-T27
8. T44-T47 and T56-T59

The fully conflicted case has 7 conflicts per phase, or 56 conflicts per
`ds_read`. Across the 64-read timing block, that saturates at 3584 conflicts
per block. The baseline latency is higher than the 32- and 64-bit reads, but
the same pattern is visible.

## Summary

On MI300X, these measurements support the expected 32-bank, 4-byte bank model:
the conflict pattern repeats every 128 bytes, and the saturated conflict count
depends on the number of LDS phases required by the access width.

| access width | phase count | lanes per phase | saturated conflicts per ds_read | saturated conflicts per 64-read block |
| ------------ | ----------- | --------------- | ------------------------------- | ------------------------------------- |
| 32-bit       | 2           | 32              | 62                              | 3968                                  |
| 64-bit       | 4           | 16              | 60                              | 3840                                  |
| 128-bit      | 8           | 8               | 56                              | 3584                                  |

All three access widths classify the LDS bank count as 32 using average thread
latency. PMC counters are displayed when available, but benchmark timer latency
is the default classification signal because counters can be unavailable or
incoherent on some ROCm/GPU combinations.
