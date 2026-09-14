# Support Matrix

## Supported In The Alpha

| Area | Status |
|---|---|
| Dense GQA weight and KV cache accounting | Supported for declared normalized facts |
| MLA MoE weight and KV cache accounting | Supported for declared normalized facts |
| FP8 MLA packed cache width | Supported only with explicit `cacheBytesPerToken` |
| Tensor parallelism | Supported when attention and KV-head divisibility checks pass |
| Data parallel request bins | Supported with whole-request bin packing |
| Expert parallel routed weights | Supported as `TP * DP`, divided once |
| Decode context parallelism | Supported when runtime declares DCP support |
| Quote units | `per_gpu_hour` and `per_node_hour` |
| Evidence status | Always `analytical_alpha` |

## Rejected Or Unknown

| Feature | Reason |
|---|---|
| Pipeline parallelism above 1 | Stage maps are not implemented |
| Unknown architectures | No generic optimistic estimator |
| Dense GQA routed weights | Dense profiles do not define MoE routing metadata |
| Non-uniform routed expert shards | Requires architecture-specific placement facts |
| Runtime throughput | Not inferable from memory accounting |
| Accuracy claims | Outside the scope of this tool |
| Cloud availability and account quota | Not checked offline |
