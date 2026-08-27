# DS4 mixed TP routed-MoE shard override

`OMLX_TP_MOE_SHARD_WEIGHTS` is a strict, default-empty override for routed
`SwitchGLU` expert banks only. It layers over a signed unequal outer TP plan;
it does not create or authorize that plan.

For the qualified M3 Ultra + M5 Max 3:5 plan, activate equal routed-MoE banks
before deployment:

```bash
export OMLX_TP_MOE_SHARD_WEIGHTS=4,4
```

The signed worker plan still installs `OMLX_TP_SHARD_WEIGHTS=3,5`. The cluster
hostfile carries the same explicit MoE value to both ranks. Resulting ownership:

| Component | M3 rank 0 | M5 rank 1 |
|---|---:|---:|
| attention / sinks / LoRA | 3/8 | 5/8 |
| shared dense expert | 3/8 | 5/8 |
| routed SwitchGLU gate/up/down | 4/8 | 4/8 |
| vocab and other outer-sharded tensors | 3/8 | 5/8 |

Unset or empty keeps the existing outer split for every tensor. The override
fails before model sharding unless it has one positive integer per TP rank,
its sum equals the signed outer vector's sum, the DS4 routed intermediate is
divisible at every boundary, and every local width preserves a 32-value MXFP4
quantization group. It is rejected when the outer split is equal or absent.

## Memory-accounting risk

The planner continues to account all layer weights using the signed outer 3:5
split. A mixed 4:4 routed bank therefore makes that estimate directionally
wrong: it underestimates M3 residency and overestimates M5 residency.

DS4 Flash routed experts are approximately 3.42 GB/layer. Across 43 layers,
moving one eighth of that bank from M5 to M3 shifts about 18.4 GB:

```text
3.42 GB/layer * 43 layers / 8 = 18.38 GB
```

The 256 GB M3 has enough measured headroom for this specific shift, while the
128 GB M5 benefits from the same reduction. This does not make arbitrary mixed
vectors safe. Until planner accounting models component-level splits, operators
must treat the M3 estimate as roughly 18.4 GB low and retain the normal memory
guard and load preflight.

The override changes parameter residency only. Collective count/order, routed
expert selection, score weighting, decode behavior, and mathematical output
remain unchanged; the two local partials still sum over the same full 2048-wide
intermediate.
