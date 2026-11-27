"""DFly2 draft-forward support for DeepSeek-V4.1 (farm patch).

oMLX ships DFly only for the Hy V4 family; DeepSeek-V4.1 has MTP instead. This
package carries the DFly2 math and the target-cache bridge needed to run the
official `AngelSlim/DS-V4.1-Flash-DFly2-B6` draft next to a DeepSeek-V4.1 target.

Nothing here is wired into the serving path yet. The contract is:

* :mod:`official_loader` validates the official checkpoint by metadata before a
  single tensor is read, so a staging mistake can never turn into silent random
  weights. It numbers draft layers from zero; ``target_layer_map`` translates to
  the target layers the draft shadows (``dfly_config.target_kv_layers``).
* :mod:`dsv41_bridge` owns the request-local view of the target cache: the
  compressed global MLA latents plus the uncompressed local window, and the
  index choices the target indexer produced for anchor-1.
* :mod:`dfly2_block` runs one draft layer; :mod:`dfly2_sparse_attention` and
  :mod:`dfly2_heads` hold its attention and projection pieces.
* :mod:`dsv41_verify` compares draft proposals with the target's own tokens and
  computes the accepted prefix, which is what a cache rollback needs.
* :mod:`omlx_v41_capture` is the seam into oMLX: capture after the target
  indexer produces ``shared["idx"]`` and before target KV is packed, and call
  ``rollback(keep_positions)`` from the same path that truncates the verify
  cache.

Versions must match the engine exactly: the draft is only valid against the
target checkpoint it was trained on, and the cache layout is part of that
contract.
"""
from __future__ import annotations

from .dfly2_block import draft_block
from .dfly2_context import DFly2ContextConfig, DFly2LayerContext, build_dfly2_context, build_layer_context
from .dfly2_heads import final_norm, hidden_correction, mtp_fusion
from .dfly2_sparse_attention import selected_target_attention
from .dsv41_bridge import DSV41History, DSV41TargetBridge
from .dsv41_verify import GreedyVerification, greedy_prefix
from .omlx_v41_capture import capture_attention_slice
from .official_loader import (
    LoadPlan,
    LoadedOfficialWeights,
    OfficialDFly2Checkpoint,
    draft_layer_ids,
    required_prefixes,
    target_layer_map,
)

__all__ = [
    "DFly2ContextConfig",
    "DFly2LayerContext",
    "DSV41History",
    "DSV41TargetBridge",
    "GreedyVerification",
    "LoadPlan",
    "LoadedOfficialWeights",
    "OfficialDFly2Checkpoint",
    "build_dfly2_context",
    "build_layer_context",
    "capture_attention_slice",
    "draft_block",
    "draft_layer_ids",
    "final_norm",
    "greedy_prefix",
    "hidden_correction",
    "mtp_fusion",
    "required_prefixes",
    "selected_target_attention",
    "target_layer_map",
]
