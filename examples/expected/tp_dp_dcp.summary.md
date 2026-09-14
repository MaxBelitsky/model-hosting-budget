# Expected Summary: TP DP DCP

Input: `examples/tp_dp_dcp.json`

The synthetic MLA MoE model declares 80.000 GB shared weights, 160.000 GB routed weights, 4.000 GB replicated weights, and 16 routed experts.

Expected residency changes:

- `TP4 DP1 DCP1`: shared 20.000 GB, routed 40.000 GB, replicated 4.000 GB.
- `TP2 DP2 DCP1`: shared 40.000 GB, routed 40.000 GB, replicated 4.000 GB.
- `TP4 DP1 DCP4`: same weights as `TP4 DP1 DCP1`, with smaller cache residency from DCP.

All predictions remain `analytical_alpha`.
