# Methodology

All memory arithmetic is in bytes. The evaluator is a deterministic accounting tool, not a GPU benchmark or runtime simulator.

## Inputs

A workload declares request groups, model profiles, targets, budgets, runtime capabilities, and quotes in a single versioned JSON document. The CLI reads that document and does not dereference paths, URLs, model IDs, provider IDs, or provenance references. Any public or private evidence must be normalized into the input by the caller.

The published JSON Schema documents the portable shape of the file. The Python runtime validator is authoritative for cross-field arithmetic constraints such as head divisibility, expert-parallel divisibility, FP8 MLA cache-width floors, exact reserve boundaries, and implemented parallelism.

## Weights

Model weights are grouped as:

- `shared`: sharded by tensor parallelism.
- `routed`: sharded by expert parallelism, where `EP = TP * DP`, and divided once.
- `replicated`: resident on every rank.

Per-rank sharded byte counts use integer ceiling division. This is intentionally conservative when a group does not divide evenly.

## KV Cache

Dense GQA profiles use:

```text
layers * 2 * kvHeads * headDim * dtypeBytes * roundedTokens / min(TP, kvHeads)
```

MLA MoE profiles use:

```text
layers * cacheBytesPerToken * roundedTokens
```

For FP8 MLA, `cacheBytesPerToken` is required because physical packed layouts can exceed the logical latent-plus-rope width. BF16 MLA may use the logical width when no packed width is supplied.

Request tokens are rounded up to 256-token blocks. Decode context parallelism divides token blocks only when runtime support is declared and the layout is otherwise compatible.

## Budgets

The per-GPU capacity budget is:

```text
gpuMemoryBytes * (1 - deviceReserveFraction)
```

Budget comparisons use exact rational arithmetic derived from decimal strings, then report the integer floor as `perGpuBudget`. This avoids verdict drift around long decimal reserve fractions.

Statuses are:

- `fits`: planned bytes are at or below the exact reserved budget.
- `tight`: planned bytes exceed the reserve but fit nominal GPU memory.
- `exceeds`: planned bytes exceed nominal GPU memory.
- `unknown`: compatibility failed before capacity arithmetic was meaningful.

## Cost

Quote amounts are exact decimal strings. Node count is `ceil(physicalGPUs / gpusPerNode)`. Per-GPU quotes are multiplied by `gpusPerNode` first, so they still reflect the declared purchasable bundle.
