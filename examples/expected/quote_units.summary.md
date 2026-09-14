# Expected Summary: Quote Units

Input: `examples/quote_units.json`

Both targets use a two-GPU layout on hardware declared as an eight-GPU purchased bundle.

Expected quote normalization:

- `GPU Metered Cloud`: `2.00` USD per GPU-hour becomes `16` USD per node-hour and `16` USD per billed hour.
- `Node Price Cloud`: `15.00` USD per node-hour remains `15` USD per billed hour.
- `Conditional Cloud`: `12.00` USD per node-hour is selected for the second target with cost status `conditional`.
- `Quote Only Cloud`: retained as `missing` and not treated as free.
