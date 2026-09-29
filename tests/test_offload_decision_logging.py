# SPDX-License-Identifier: Apache-2.0
"""A stable forced-offload decision must not be logged once per status read.

``_qwen4_ple_offload_status`` and ``_deepseek_v41_engram_offload_status`` are
reached from read-only paths as well: the admin model list, the per-model
detail endpoint and the settings-signature projection. Those run once per UI
refresh, so warning on every call turns one stable decision into thousands of
identical log lines (measured with a desktop app polling the model list every
~2s: ~1.7k lines/hour). The decision itself is unchanged; only its emission is
deduplicated, and a real change of the decision still reports.
"""

import json
import logging
import struct

from omlx.engine_pool import EngineEntry, EnginePool
from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
    qwen4_exp_residency_estimate,
)

FORCED_WARNING = "Qwen4-Exp PLE forced to SSD"
ENGINE_POOL_LOGGER = "omlx.engine_pool"


def _write_safetensors(path, tensors: dict[str, int]) -> None:
    offset = 0
    header = {}
    for key, size in tensors.items():
        header[key] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def _qwen4_fixture(tmp_path):
    """A tiny qwen4_exp checkpoint whose resident and mmap sizes differ."""

    model = tmp_path / "qwen4"
    model.mkdir()
    ple_key = "model.language_model.ngram_embedding.shard_0.weight"
    _write_safetensors(model / "model.safetensors", {ple_key: 100})
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {ple_key: "model.safetensors"}})
    )
    estimate = qwen4_exp_residency_estimate(model)

    pool = EnginePool.__new__(EnginePool)
    pool._get_admission_ceiling = None
    pool._get_admission_soft_target = None
    pool._get_final_ceiling = None

    entry = EngineEntry.__new__(EngineEntry)
    entry.model_id = "qwen4"
    entry.model_path = str(model)
    entry.config_model_type = "qwen4_exp"
    return pool, entry, estimate


def _forced_warnings(caplog) -> int:
    return sum(FORCED_WARNING in record.getMessage() for record in caplog.records)


def test_polled_status_reads_warn_once(tmp_path, caplog):
    """Five identical reads must not produce five identical warnings."""

    pool, entry, estimate = _qwen4_fixture(tmp_path)
    between_modes = (estimate.resident_bytes + estimate.mmap_bytes) // 2
    pool._get_residency_ceiling = lambda: between_modes

    with caplog.at_level(logging.WARNING, logger=ENGINE_POOL_LOGGER):
        results = [pool._qwen4_ple_offload_status(entry, None) for _ in range(5)]

    assert all(forced is True for _, forced, _ in results)
    assert _forced_warnings(caplog) == 1


def test_transition_back_to_forced_is_reported_again(tmp_path, caplog):
    """Deduplication must not swallow a real change of the decision."""

    pool, entry, estimate = _qwen4_fixture(tmp_path)
    between_modes = (estimate.resident_bytes + estimate.mmap_bytes) // 2
    ceiling = [between_modes]
    pool._get_residency_ceiling = lambda: ceiling[0]

    with caplog.at_level(logging.WARNING, logger=ENGINE_POOL_LOGGER):
        pool._qwen4_ple_offload_status(entry, None)
        ceiling[0] = estimate.resident_bytes
        pool._qwen4_ple_offload_status(entry, None)
        ceiling[0] = between_modes
        pool._qwen4_ple_offload_status(entry, None)

    assert _forced_warnings(caplog) == 2


def test_resident_model_never_warns(tmp_path, caplog):
    """A model that fits must stay silent, however often it is read."""

    pool, entry, estimate = _qwen4_fixture(tmp_path)
    pool._get_residency_ceiling = lambda: estimate.resident_bytes

    with caplog.at_level(logging.WARNING, logger=ENGINE_POOL_LOGGER):
        results = [pool._qwen4_ple_offload_status(entry, None) for _ in range(3)]

    assert all(forced is False for _, forced, _ in results)
    assert _forced_warnings(caplog) == 0
