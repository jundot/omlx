import os
import struct
import subprocess
import sys
import zlib

# MLX 0.32.2 runs fp32 GPU matmuls at TF32 precision on M5-class tensor units;
# the fp32 parity tests assert 2e-5, which TF32 cannot hold. Test session only.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration and fixtures for oMLX tests.

This module provides common fixtures used across test files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# Install the torch stub before any test imports xgrammar (e.g. via @patch
# decorators that resolve the target at collection time). When real torch is
# present this is a no-op; in the DMG layout it satisfies xgrammar's
# import-time torch references so the package can load.
from omlx._torch_stub import install as _install_torch_stub
_install_torch_stub()

# Run tests under the same M5 sorted gather_qmm reroute the server
# installs at model load (issue #2267). Without it, kernel-sensitive
# tests (e.g. the SwitchGLU fusion bit-exactness test, whose inter=32
# down_proj runs at K=32) fail on M5 hardware. No-op elsewhere.
from omlx.patches.m5_gather_qmm import apply_m5_gather_qmm_workaround
apply_m5_gather_qmm_workaround()

from omlx.request import Request, SamplingParams

# Tests that keep the GPU busy for seconds on the CI runner. Under --dist
# loadgroup they share one worker so concurrent command buffers do not trip the
# Metal GPU timeout. Every other test runs with the rest of its file. Add a test
# here when CI reports kIOGPUCommandBufferCallbackErrorTimeout for it.
_GPU_SERIAL_TESTS = {
    "test_deepseek_v41.py": {
        "test_chunk_boundaries_and_late_join",
        "test_in_place_cache_growth_matches_concatenation",
        "test_official_prefill_decode",
        "test_persistent_cache_is_packed_and_decode_only_unpacks_selected_rows",
        "test_real_vlm_engine_text_and_images",
        "test_vision_and_aligner_reference",
    },
    "test_deepseek_v41_fused_decode.py": {
        "test_batched_decode_matches_single_requests",
        "test_batched_mma_kernels_match_rows_kernels",
        "test_batched_moe_matches_single_token_moe",
        "test_fused_decode_matches_unfused_block",
        "test_mtp_generation_matches_greedy_with_fused_kernels",
        "test_offloaded_experts_decode_on_the_unfused_path",
        "test_verify_block_matches_token_by_token_decode",
    },
    "test_deepseek_v41_prefill_backpressure.py": {
        "test_v41_prefill_evals_stream_per_layer",
    },
    "test_deepseek_v4_batch_padding.py": {
        "test_short_request_after_long_one_leaves_batch",
    },
    "test_deepseek_v4_dspark.py": {
        "test_dspark_target_tap_and_parallel_draft_shapes",
    },
    "test_deepseek_v4_offload.py": {
        "test_wrap_and_release_return_descriptors_to_baseline",
        "test_wrap_replaces_module_and_keeps_only_slots",
    },
    "test_glm5_next_mtp.py": {
        "test_vector_restore_and_continuation_match_scalar",
    },
    "test_glm_moe_dsa_offload.py": {
        "test_wrap_and_release_return_descriptors_to_baseline",
    },
    "test_mlx_lm_mtp_patch.py": {
        "test_context_copy_drafts_keep_greedy_output",
        "test_copied_drafts_preserve_filtered_target_marginal",
        "test_multi_request_mtp_or_singleton_only_matches_standard",
        "test_qwen_late_join_as_batch_row_finishes_skips_history_replay",
        "test_shared_verify_boundary_emit_uses_private_row_cache",
        "test_singleton_mtp_rebinds_positions_after_prefill_clear",
        "test_sparse_stochastic_acceptance_preserves_filtered_target_marginal",
        "test_stochastic_acceptance_preserves_target_marginal",
        "test_vector_commit_matches_scalar_states_and_ordinary_tokens",
    },
    "test_mlx_vlm_glm5_next_compat.py": {
        "test_bitwise_equal_to_canonical_path_without_tf32",
        "test_sparse_attention_native_routes_get_fp16_despite_fp32_activations",
    },
    "test_mlx_vlm_qwen4_exp_compat.py": {
        "test_qwen4_fp16_model_stays_fp16_with_bf16_ple_table",
    },
    "test_moe_expert_offload.py": {
        "test_overlap_matches_serial_cache_state",
    },
    "test_qwen35_moe_routed_decode.py": {
        "test_bf16_shared_expert_stays_composed_and_bit_identical",
        "test_experts_past_the_bound_view_are_read_from_the_stacked_weights",
        "test_tied_router_logits_route_like_the_served_block",
    },
    "test_qwen35_moe_router.py": {
        "test_fused_combine_sigmoid_matches_every_gate_encoding",
    },
    "test_qwen35_moe_verify_window.py": {
        "test_batched_decode_rows_equal_one_token_decode",
        "test_kernel_failure_falls_back_to_the_composed_verifier",
        "test_router_ties_and_high_experts",
        "test_top8_window_and_batched_rows_equal_one_token_decode",
        "test_window_rows_equal_one_token_decode",
    },
    "test_qwen35_verify_sdpa_split.py": {
        "test_chain_tail_matches_cache_writes",
    },
    "test_qwen4_batch_qsa_indexer.py": {
        "test_indexer_capacity_matches_concatenate_bit_for_bit",
    },
    "test_qwen4_hc_fused.py": {
        "test_decode_and_verify_rows_match_eager_writes",
        "test_prefill_chunks_match_eager_writes",
    },
    "test_qwen4_qsa_reserved_capacity.py": {
        "test_scheduler_reserves_kv_once_and_decode_grows_in_steps",
    },
    "test_qwen4_verify_attention_rows.py": {
        "test_masked_verify_rollback_then_decode_matches_serial",
        "test_masked_verify_rows_equal_serial_decode_steps",
    },
    "test_row_exact_qmv.py": {
        "test_rows_equal_one_row_quantized_matmul",
    },
    "test_sdpa256_attention.py": {
        "test_flash_sdpa256_memory_is_sub_quadratic",
    },
}


def _gpu_serial(item) -> bool:
    name = getattr(item, "originalname", item.name)
    return name in _GPU_SERIAL_TESTS.get(item.path.name, ())


@pytest.hookimpl(tryfirst=True)  # Before xdist reads the xdist_group markers.
def pytest_collection_modifyitems(config, items):
    if not config.pluginmanager.hasplugin("xdist"):
        return
    for item in items:
        group = "gpu" if _gpu_serial(item) else item.nodeid.split("::", 1)[0]
        item.add_marker(pytest.mark.xdist_group(group))
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # With --no-loadscope-reorder, groups start in collection order; start
        # the long serial group first.
        items.sort(key=lambda item: not _gpu_serial(item))


@pytest.fixture
def glm5_fused_decode():
    """GLM-5.3's fused decode/verify kernels, which run (and replay the
    reference bit for bit) on M3 and newer GPUs."""
    from omlx.patches.mlx_vlm_glm5_next_compat import (
        apply_mlx_vlm_glm5_next_compat_patch,
    )

    apply_mlx_vlm_glm5_next_compat_patch()
    from mlx_vlm.models.glm5_next import language

    if not language._DECODE_FUSION:
        pytest.skip("the fused GLM-5.3 decode kernels run on M3 and newer GPUs")
    return language


@pytest.fixture
def apply_drafter_target(monkeypatch):
    """``drafter_target.apply`` that restores the patched classes afterwards.

    The patch rewrites shared mlx-lm/mlx-vlm model and layer classes, which
    other tests in the session build too.
    """
    from omlx.patches.mlx_lm_mtp import drafter_target

    model_names = (
        "__call__",
        "mtp_partial_rollback",
        "rollback_speculative_cache",
        "_omlx_drafter_single_row",
        "_omlx_drafter_target",
    )
    layer_names = ("__call__", "_omlx_drafter_capture_wrapped")

    def apply(model):
        targets = [(type(model), model_names)]
        targets += [
            (cls, layer_names) for cls in {type(layer) for layer in model.model.layers}
        ]
        for cls, names in targets:
            for name in names:
                if name in cls.__dict__:
                    monkeypatch.setattr(cls, name, cls.__dict__[name])
                else:
                    # Records the name as absent so teardown deletes it.
                    monkeypatch.setattr(cls, name, None, raising=False)
                    delattr(cls, name)
        return drafter_target.apply(model)

    return apply


@pytest.fixture(autouse=True)
def cluster_home(tmp_path, monkeypatch):
    from omlx.cluster import ssh_keys, worker_shim

    home = tmp_path / "cluster-home"
    publish = worker_shim.ensure_cluster_python_shim

    def publish_shim(**kwargs):
        if kwargs.get("home") is None:
            kwargs["home"] = home
        return publish(**kwargs)

    monkeypatch.setattr(worker_shim, "ensure_cluster_python_shim", publish_shim)
    # SSH paths are resolved at import time, before test fixtures run.
    ssh_dir = home / ".ssh"
    monkeypatch.setattr(ssh_keys, "_SSH_DIR", ssh_dir)
    monkeypatch.setattr(ssh_keys, "_SSH_KEY_PATH", ssh_dir / "omlx_cluster")
    monkeypatch.setattr(ssh_keys, "_SSH_PUBKEY_PATH", ssh_dir / "omlx_cluster.pub")
    return home


class MockTokenizer:
    """Mock tokenizer for testing without loading real models."""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids (simple simulation)."""
        # Simple simulation: each word becomes a token
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        # Simulate tokenization by splitting on spaces
        for i, word in enumerate(text.split()):
            # crc32, unlike hash(), is stable across processes.
            token_id = (zlib.crc32(word.encode()) % (self.vocab_size - 10)) + 10
            tokens.append(token_id)
        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token ids to text (simple simulation)."""
        if skip_special_tokens:
            token_ids = [
                t
                for t in token_ids
                if t not in (self.eos_token_id, self.pad_token_id, self.bos_token_id)
            ]
        # Return a placeholder string representing the token count
        return f"<decoded:{len(token_ids)} tokens>"

    def __call__(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Tokenize text and return dict with input_ids."""
        input_ids = self.encode(text)
        return {"input_ids": input_ids}


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(
        self,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        vocab_size: int = 32000,
        model_type: str = "llama",
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.vocab_size = vocab_size
        self.model_type = model_type


class MockModel:
    """Mock model for testing without loading real models."""

    def __init__(self, config: Optional[MockModelConfig] = None):
        self.config = config or MockModelConfig()
        self._parameters: Dict[str, Any] = {}

    def __call__(self, input_ids: Any, **kwargs: Any) -> Any:
        """Forward pass (returns mock logits)."""
        mock_output = MagicMock()
        mock_output.shape = (1, len(input_ids) if hasattr(input_ids, "__len__") else 1, self.config.vocab_size)
        return mock_output

    def parameters(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self._parameters

    def make_cache(self) -> list:
        """Build the per-layer prompt cache, like a real mlx-lm model.

        The scheduler probes this to decide whether a stored prefix can be
        rebuilt faithfully, so the double has to answer it. A plain llama-style
        model builds ``KVCache`` layers; tests that need another cache class
        override this attribute.
        """
        from mlx_lm.models.cache import KVCache

        return [KVCache() for _ in range(self.config.num_hidden_layers)]


@pytest.fixture
def mock_tokenizer() -> MockTokenizer:
    """Provide a mock tokenizer for tests."""
    return MockTokenizer()


@pytest.fixture
def mock_cluster_ssh(monkeypatch):
    from omlx.cluster import launch

    runner = MagicMock(
        return_value=subprocess.CompletedProcess(
            [], 0, stdout='{"action": "no-marker"}', stderr=""
        )
    )
    monkeypatch.setattr(launch, "_run_cluster_ssh", runner)
    return runner


@pytest.fixture
def mock_model() -> MockModel:
    """Provide a mock model for tests."""
    return MockModel()


@pytest.fixture
def mock_model_config() -> MockModelConfig:
    """Provide a mock model configuration for tests."""
    return MockModelConfig()


def _gguf_value(value) -> bytes:
    if isinstance(value, int):
        return struct.pack("<II", 4, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    if isinstance(value, str):
        data = value.encode()
        return struct.pack("<IQ", 8, len(data)) + data
    if all(isinstance(item, str) for item in value):
        items = b"".join(
            struct.pack("<Q", len(item.encode())) + item.encode() for item in value
        )
        return struct.pack("<IIQ", 9, 8, len(value)) + items
    if value and all(isinstance(item, bool) for item in value):
        items = b"".join(struct.pack("<?", item) for item in value)
        return struct.pack("<IIQ", 9, 7, len(value)) + items
    items = b"".join(struct.pack("<i", item) for item in value)
    return struct.pack("<IIQ", 9, 5, len(value)) + items


@pytest.fixture
def write_gguf():
    """Write a GGUF file of a header with ``metadata`` and ``size`` bytes in all."""

    def write(path: Path, metadata: dict, size: int = 0) -> Path:
        header = b"GGUF" + struct.pack("<IQQ", 3, 0, len(metadata))
        for key, value in metadata.items():
            header += struct.pack("<Q", len(key)) + key.encode() + _gguf_value(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header + b"\0" * max(0, size - len(header)))
        return path

    return write


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Provide a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def sample_request() -> Request:
    """Factory fixture for creating sample Request objects."""
    return Request(
        request_id="test-request-001",
        prompt="Hello, world!",
        sampling_params=SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
        ),
    )


@pytest.fixture
def sample_request_factory():
    """Factory fixture for creating multiple Request objects."""

    def _create_request(
        request_id: str = "test-request-001",
        prompt: str = "Hello, world!",
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Request:
        return Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
        )

    return _create_request


@pytest.fixture
def real_model_dir() -> Path:
    """Return the path to real models directory.

    Note: Tests using this fixture may require actual model files
    and should be marked with @pytest.mark.slow.
    """
    return Path.home() / "Workspace" / "models"


@pytest.fixture(autouse=True)
def _reset_decode_activity_registry():
    """Keep the process-global decode-activity registry hermetic per test.

    Schedulers publish to it from step(); entries live for a short TTL, so
    without this a scheduler stepped in one test reads as cross-engine
    decode contention in the next.
    """
    from omlx.decode_activity import get_decode_activity

    get_decode_activity().clear()
    yield
    get_decode_activity().clear()


@pytest.fixture(autouse=True)
def _reset_vlm_mtp_attach():
    """Start each test with the process-global VLM MTP attach flag at its default.

    VLM pre-load patches set it from each checkpoint, so a test that builds a
    patched model directly would otherwise inherit an earlier test's value.
    """
    mtp = sys.modules.get("omlx.patches.mlx_vlm_mtp")
    if mtp is not None:
        mtp.set_mtp_attach_enabled(True)
    yield
    if mtp is not None:
        mtp.set_mtp_attach_enabled(True)


@pytest.fixture(autouse=True)
def _reset_batched_decode_qmm():
    """Keep the process-global batched decode routing off between tests.

    Pre-load patches enable it, and it changes multi-row decode numerics for
    every later test in the worker.
    """
    module = sys.modules.get("omlx.patches.batched_decode_qmm")
    if module is not None:
        module.remove_batched_decode_qmm_patch()
    yield
    module = sys.modules.get("omlx.patches.batched_decode_qmm")
    if module is not None:
        module.remove_batched_decode_qmm_patch()


@pytest.fixture(autouse=True)
def _reset_api_key_failures():
    """Keep the process-global wrong API key budget hermetic per test."""
    auth = sys.modules.get("omlx.admin.auth")
    if auth is not None:
        auth._key_failures.clear()
    yield
    auth = sys.modules.get("omlx.admin.auth")
    if auth is not None:
        auth._key_failures.clear()


@pytest.fixture(autouse=True)
def _hermetic_metal_release_accounting(monkeypatch):
    """Keep the graphics footprint and its release-lag history test-local.

    Tests feed synthetic phys_footprint values. The test process's real
    graphics ledger and earlier tests' samples would otherwise split or
    discount them unpredictably. Tests of the split patch these explicitly.
    """
    for name in (
        "omlx.scheduler",
        "omlx.process_memory_enforcer",
        "omlx.utils.metal_sync",
    ):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "get_graphics_footprint"):
            monkeypatch.setattr(module, "get_graphics_footprint", lambda: 0)
    metal_sync = sys.modules.get("omlx.utils.metal_sync")
    if metal_sync is not None:
        metal_sync._residuals.clear()
        metal_sync._last_unreleased = (0.0, 0)
    yield
    if metal_sync is not None:
        metal_sync._residuals.clear()
        metal_sync._last_unreleased = (0.0, 0)
