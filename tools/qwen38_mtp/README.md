# Qwen3.8-27B native-MTP port tooling

Tooling for the `perf/mlx-fast-27b` port of the
[Layr-Labs/qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)
optimizations into omlx's Lightning MTP path. The correctness gate is
`tests/test_qwen38_mtp_token_exact.py`; these scripts stage the ~16 GB
artifacts and drive decode-level comparisons.

## Staging the pinned checkpoints (~/qwen38-mtp)

The challenge pins the artifacts by digest:

| Artifact | Repo / revision | Records |
|---|---|---|
| backbone | `EigenLabs/Qwen3.8-27B-4bit` @ `eda45ab47f465d08d6558f0353a2346e2eb9d5b3` | 10 |
| head | `EigenLabs/Qwen3.8-27B-MTP-bf16` @ `26a328e070875b0314d652a039b6b59902690f03` | 4 |

```bash
git clone https://github.com/Layr-Labs/qwen-3.8-mtp-challenge ~/qwen38-mtp/challenge
cd ~/qwen38-mtp && ./download.sh     # snapshot_download + sha256 verify vs fixtures
python tools/qwen38_mtp/merge_checkpoint.py   # backbone + mtp.-prefixed head -> ~/qwen38-mtp/merged
```

The merge contract mirrors the Swift loader: the head's 15 bare tensors are
prefixed `mtp.` and merged into the backbone tree (written as an extra
`model-00004-of-00004.safetensors` shard so mlx-lm's `model*.safetensors`
glob loads them).

## Token-exactness

```bash
python tools/qwen38_mtp/bg_harness.py serial 96     # serial reference trajectory
python tools/qwen38_mtp/bg_harness.py mtp 96 2      # MTP decode at draft depth 2
python tools/qwen38_mtp/bg_harness.py compare mtp_d2.json
# -> TOKEN-EXACT iff every emitted token equals the serial trajectory
```

One model load per process — never run two invocations concurrently on a
64 GB machine (each process holds ~15 GB).

## Port-log

Per-submission status lives in `docs/qwen-mtp-port-log.md`, updated with
every submission's port commit (labeled with the submission UUID).
