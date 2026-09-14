# Expected Summary: KV Failure

Input: `examples/kv_failure.json`

The synthetic dense GQA model has 10.000 GB of per-rank weights on a 40.000 GB GPU. Its single 7,000,064-token request produces 114.689 GB of KV cache after 256-token block accounting. With a 4.000 GB runtime reserve, planned per-rank memory is 128.689 GB.

Expected statuses:

- compatibility: `pass`
- capacity: `exceeds`
- evidence: `analytical_alpha`
- cost: `no_qualifying_quote`

The example demonstrates a model whose weights alone fit device memory while the declared workload does not.
