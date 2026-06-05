# Rocprof LDS Counter Notes

This note records the initial LDS counter investigation for the phase-group
harness. Experiments run on shark-mkm-1:

- GPU target: `gfx1201`
- Visible GPU agents: two `gfx1201` agents from `rocm_agent_enumerator`
- `rocprofv3` path after activation:
  `venv/bin/rocprofv3`
- `rocprofv3` version: `1.3.1`
- ROCm version reported by `rocprofv3`: `7.14.0`

All rocprof commands for this project should be run from the activated virtual
environment so the venv-installed `rocprofv3` is used instead of the system
binary.

## Counter Meanings

### `SQC_LDS_BANK_CONFLICT`

`SQC_LDS_BANK_CONFLICT` is the lowest-level relevant metric found in the local
counter list:

```text
Counter_Name        : SQC_LDS_BANK_CONFLICT
Description         : Number of cycles LDS is stalled by bank conflicts. (emulated, C1)
Block               : SQ
```

### `SQC_LDS_IDX_ACTIVE`

`SQC_LDS_IDX_ACTIVE` is the denominator used by the derived percentage metric:

```text
Counter_Name        : SQC_LDS_IDX_ACTIVE
Description         : Number of cycles LDS is used for indexed (non-direct,non-interpolation) operations. {per-simd, emulated, C1}
Block               : SQ
```

### `LDSBankConflict`

`LDSBankConflict` is a derived metric:

```text
Counter_Name        : LDSBankConflict
Description         : The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0% (optimal) to 100% (bad).
Expression          : 100*reduce(SQC_LDS_BANK_CONFLICT,sum)/reduce(SQC_LDS_IDX_ACTIVE,sum)
```
