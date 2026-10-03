# Expert offload staging

Read the repository README, pyproject.toml, pytest.ini and CI workflow first.
This optional C++ transport preserves the existing read/install/cache pipeline.
Do not merge tensor reads, move cache writes to workers, add fences, or relax
bit-exact comparisons. Run GPU tests and model benchmarks sequentially.

Build and verify with the active repository environment:

```sh
OMLX_WITH_MOE_STAGING=1 python -m pip install -e ".[mcp]"
python -m pytest tests/test_moe_staging.py tests/test_moe_expert_offload.py -q
python -m pytest tests/ -m "not slow and not integration" --durations=50
```

The build requires the pinned MLX/nanobind ABI and CMake, but no Metal shader
compiler. Runtime opt-in is `OMLX_MOE_OFFLOAD_STAGING=1`; unset or `0` is the
kill switch, effective on model reload. See `docs/moe-offload-staging.md`.
