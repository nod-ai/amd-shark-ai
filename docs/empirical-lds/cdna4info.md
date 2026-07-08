# CDNA4 MI350X LDS Measurements

Measurements were collected on MI350X (`gfx950`), which has a wavefront size of
64 threads. The LDS bank count was measured to be 64. With 4-byte banks, the
bank mapping repeats every 256 bytes, so an access stride of 256 bytes causes
the maximum possible number of bank conflicts.

The captured bank-count sweep in this note covers `ds_read_b64`. No usable
normalized rocprof bank-conflict counter values were recorded for this capture,
so the table uses benchmark timer values. Latencies are average active-thread
timer values from `clock64()`, reported in cycles, and were collected using the
macro payload with 64 `ds_read` instructions. Phase groups are discovered by
comparing pairwise latencies of threads mapping to the same bank. Threads in the
same access phase experience bank conflicts.

## ds_read_b32 Phase Group

For `ds_read_b32`, the access happens in a single phase: T0-T63.

## ds_read_b64 Latencies

| stride (bytes) | avg thread latency (cycles) |
| -------------- | --------------------------- |
| 8              | 583.353                     |
| 16             | 819.207                     |
| 32             | 1490.275                    |
| 64             | 2758.733                    |
| 128            | 5089.500                    |
| 256            | 10024.566                   |
| 512            | 10014.062                   |

For `ds_read_b64`, the access happens in two phases of 32 lanes each: T0-T31,
then T32-T63. The high-latency bucket starts at the 256-byte stride and remains
saturated for the 512-byte stride.

## ds_read_b128 Phase Groups

For `ds_read_b128`, the access happens in four phases of 16 lanes each.

1. T0-T3, T12-T15, T20-T23, T24-T27
2. T32-T35, T44-T47, T52-T55, T56-T59
3. T4-T7, T8-T11, T16-T19, T28-T31
4. T36-T39, T40-T43, T48-T51, T60-T63

## Summary

On MI350X, the recorded `ds_read_b64` sweep supports a 64-bank, 4-byte bank
model: the latency pattern saturates at 256-byte stride, and larger strides stay
in the same high-latency bucket.

| access width | phase count | lanes per phase | most likely bank count |
| ------------ | ----------- | --------------- | ---------------------- |
| 32-bit       | 1           | 64              | 64                     |
| 64-bit       | 2           | 32              | 64                     |
| 128-bit      | 4           | 16              | 64                     |

The profiler classified the recorded sweep using average thread latency. The
CDNA4 phase groups above follow the 64-bank MI350 LDS behavior: a 32-bit read
can cover the whole 64-lane wave in one phase, while 64- and 128-bit reads split
the wave into two and four phases respectively.
