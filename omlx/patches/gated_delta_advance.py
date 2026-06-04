# SPDX-License-Identifier: Apache-2.0
"""No-op as of mlx-vlm 041f889 -- the GatedDeltaNet fixes are now upstream.

History
-------
This patch used to replace mlx-vlm ``Qwen3_5GatedDeltaNet.__call__`` to carry
two fixes upstream lacked:

- ``9dcefa5`` "break shared-buffer memory leak in GatedDeltaNet cache" -- wrap
  the ``cache[0]`` write in ``mx.contiguous`` and add the ``cache.lengths is
  not None`` per-element slicing branch.
- Drop the mlx-vlm silent fallbacks (``conv_state.shape[0] != B`` => zeros,
  same shape for state and mask) that mask real bugs.

As of mlx-vlm 041f889 BOTH motivations are gone:

1. The ``mx.contiguous(cache[0])`` write and the ``cache.lengths`` /
   ``take_along_axis`` per-element branch are now in upstream verbatim
   (Qwen3_5GatedDeltaNet.__call__, mlx-vlm 041f889). Redundant.

2. The ``conv_state.shape[0] != B`` and ``mask.shape[0] != B`` fallbacks this
   patch used to DROP are now load-bearing: 041f889's batched ``target_verify``
   (speculative-decode verify) path produces ragged-row shapes that rely on
   those guards. Dropping them would BREAK spec-decode.

3. Upstream's GatedDeltaNet is now a strict superset of the old body: it adds
   ``target_verify``, ``_gated_delta_update_verify_decode``, a 12-element
   ``gdn_sink`` tuple ending in ``intermediate_states``, and the
   ``_causal_conv1d_verify`` / ``_causal_conv1d_decode`` fast paths. The 11-tuple
   this patch's old body appended is consumed by mlx-vlm's stock
   ``LanguageModel.rollback_speculative_cache`` -- which now expects the 12-tuple.
   Installing the old body would feed a stale tuple to stock rollback and break
   speculative cache rollback for the VLM MTP path.

So overriding ``__call__`` is now both pointless and actively harmful. The patch
is therefore a deliberate no-op: ``apply_gated_delta_advance_patch`` returns
False without touching the class, leaving 041f889's GatedDeltaNet intact. The
file is kept (not deleted) so the import sites in ``omlx/engine/vlm.py`` and
``omlx/utils/model_loading.py`` stay valid and the decision is documented here.

NOTE (flyto soft-fork): upstream jundot/omlx deleted this file in 3d2fef5 when
it bumped mlx-vlm. flyto keeps the no-op shell rather than deleting, so the
companion ``qwen3_5_attention.py`` correctness patch (which flyto retains and
upstream dropped) and its apply call-sites remain undisturbed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_gated_delta_advance_patch(model: Any = None) -> bool:
    """No-op as of mlx-vlm 041f889. See module docstring for why.

    Kept as a stable entry point so the call sites in
    ``omlx/engine/vlm.py`` and ``omlx/utils/model_loading.py`` need no
    change while flyto soft-forks. Overriding mlx-vlm 041f889's
    ``Qwen3_5GatedDeltaNet.__call__`` is now both redundant (its fixes are
    upstream) and unsafe (it would clobber the new ``target_verify`` path and
    feed a stale 11-tuple to stock ``rollback_speculative_cache``, which now
    expects a 12-tuple with ``intermediate_states``).

    The ``model`` argument is accepted for backward compatibility but not
    used. Always returns False -- no class is patched.
    """
    logger.debug(
        "gated_delta_advance patch is a no-op on mlx-vlm 041f889 "
        "(fixes upstreamed; override would break target_verify rollback)"
    )
    return False
