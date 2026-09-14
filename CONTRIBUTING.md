# Contributing

Contributions should keep the project offline, deterministic, and narrow.

For code changes, include focused tests that exercise the changed accounting or validation behavior. For model/runtime/profile changes, include a pinned source, a supported runtime capability declaration, and either an independent hand fixture or a structured observation. Do not add generic fallback estimates for unknown architectures.

Before opening a pull request, run:

```bash
python -m pytest
python -m build
```

Do not include secrets, private workloads, cloud account screenshots, model weights, bulk provider pages, or benchmark datasets.
