# Model Hosting Budget

Model Hosting Budget is an offline analytical-alpha CLI for checking whether a declared model profile, workload, runtime capability set, GPU layout, memory budget, and dated quote can form a plausible self-hosting budget.

It does not download model weights, contact cloud providers, provision hardware, estimate throughput, or claim real-model accuracy. The bundled examples are synthetic fixtures for reviewing the arithmetic.

## Quick Start

```bash
python -m pip install -e .
hosting-budget evaluate examples/quote_units.json --offline --out report/
hosting-budget explain report/result.json --model synthetic-small-gqa
hosting-budget compare report/result.json report/result.json
```

`evaluate` writes deterministic `result.json`, `result.csv`, `report.md`, and `index.html` files. `--offline` is required in this MVP.

## What It Checks

The evaluator keeps four statuses separate:

- `compatibility`: whether the declared model architecture, runtime capabilities, and parallel layout are supported by this alpha.
- `capacity`: whether per-rank weights, worst request-bin KV cache, runtime reserve, and device reserve fit the declared GPU memory.
- `evidence`: always `analytical_alpha` in the MVP. Attached observations are preserved as context but are not promoted into measured predictions.
- `cost`: whether a qualifying capacity result has priced, conditional, missing, or no quote support.

Supported model profile types are `dense_gqa` and `mla_moe`. Unknown architectures are rejected by the v1 schema, and unsupported features return failed compatibility rather than optimistic estimates. Arbitrary model names are accepted when the input supplies complete supported typed facts and provenance.

## Parallelism

Physical GPUs are computed as:

```text
physicalGPUs = tensorParallel * dataParallel * pipelineParallel
```

Pipeline parallelism is rejected unless `pipelineParallel` is `1`; stage maps are not implemented yet. For MoE profiles, `expertParallel = tensorParallel * dataParallel`, and routed weights are divided by that value exactly once, using conservative ceiling bytes per rank. Request data parallelism uses whole-request bins; a request is never split across DP ranks. Decode context parallelism is explicit and divides cache token blocks for supported layouts.

## Quotes

Quotes are dated input records. `per_node_hour` quotes bill by whole purchased nodes. `per_gpu_hour` quotes are also normalized through `hardware.gpusPerNode`; set `gpusPerNode` to `1` only when the provider actually sells individual GPUs. Missing and conditional quotes are preserved in output and are not silently treated as free.

## Examples

- `examples/kv_failure.json`: weights fit, but KV cache and runtime reserve exceed the GPU.
- `examples/tp_dp_dcp.json`: TP, DP, EP, and DCP change per-rank residency.
- `examples/quote_units.json`: per-GPU and per-node quotes normalize to whole-node bills.

See `docs/methodology.md`, `docs/support-matrix.md`, and `schemas/workload-v1.json` for the public input contract.
