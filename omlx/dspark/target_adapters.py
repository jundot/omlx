"""Target tap/rollback adapters used by the native dSpark provider.

The registry adapts an already-loaded oMLX target wrapper.  It never loads a
model and never installs a process-wide runtime facade.  Qwen3.5/3.6's oMLX
kernel patch closes over ``gated_delta_update``; recording ``_process_chunk``
inputs is therefore the authoritative way to rebuild recurrent state after a
partially rejected verify window.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from types import MethodType
from typing import Any, Protocol

logger = logging.getLogger(__name__)
_CAPTURE_LOCK = threading.RLock()


@contextmanager
def target_capture_lock():
    """Serialize the brief class/module hook window across loaded engines."""
    with _CAPTURE_LOCK:
        yield


class TargetTapAdapter(Protocol):
    family: str

    def matches(self, target: Any) -> bool: ...
    def install(self, target: Any) -> bool: ...


_QWEN_TYPES = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_6",
    "qwen3_6_text",
    "qwen3_6_moe",
    "qwen3_6_moe_text",
}


def _model_type(target: Any) -> str:
    model = getattr(target, "model", target)
    inner = getattr(model, "language_model", model)
    args = getattr(model, "args", None) or getattr(inner, "args", None)
    return str(getattr(args, "model_type", "") or "").lower()


class Qwen35TapAdapter:
    family = "qwen3.5/qwen3.6"

    def matches(self, target: Any) -> bool:
        return _model_type(target) in _QWEN_TYPES

    def install(self, target: Any) -> bool:
        modules = list(getattr(target, "_gdn_modules", ()) or ())
        if not getattr(target, "is_hybrid", False) or not modules:
            # Dense Qwen variants only need ordinary KV trim, already supplied
            # by native_target.Target.
            return True
        gdn_cls = type(modules[0])
        if not getattr(gdn_cls.__call__, "_omlx_mtp_call_marker", False):
            # Stock mlx-lm is handled by Target's native gated-delta capture.
            return True
        original_process = getattr(gdn_cls, "_process_chunk", None)
        if not callable(original_process):
            raise RuntimeError(
                "oMLX Qwen3.5/3.6 kernel lacks _process_chunk; exact dSpark "
                "rollback cannot be guaranteed"
            )
        module_ids = {id(module) for module in modules}

        @contextmanager
        def capture_linear(self: Any):
            records: list[tuple[Any, ...]] = []

            def recording_process(
                module: Any,
                qkv: Any,
                a: Any,
                b: Any,
                conv_state: Any,
                ssm_state: Any,
                ssm_mask: Any = None,
                lengths: Any = None,
            ) -> Any:
                if id(module) in module_ids:
                    records.append(
                        (module, qkv, a, b, conv_state, ssm_state, ssm_mask, lengths)
                    )
                return original_process(
                    module, qkv, a, b, conv_state, ssm_state, ssm_mask, lengths
                )

            with target_capture_lock():
                gdn_cls._process_chunk = recording_process
                try:
                    yield records, []
                finally:
                    gdn_cls._process_chunk = original_process

        def rollback(
            self: Any,
            cache: Any,
            n_rejected: int,
            accepted: list[int],
            *,
            capture: Any = None,
            spec_width: int | None = None,
        ) -> None:
            del accepted
            if n_rejected <= 0:
                if capture is None:
                    self._stash = None
                return
            stash = self._stash if capture is None else capture
            width = self._spec_width if spec_width is None else spec_width
            if stash is None:
                raise RuntimeError("dSpark rollback has no preceding verify capture")
            records = stash[0]
            keep = width - n_rejected
            linear_caches = [item for item in cache if not self._is_trimmable(item)]
            if len(records) != len(linear_caches):
                raise RuntimeError(
                    "dSpark GatedDeltaNet rollback capture mismatch: "
                    f"{len(records)} records for {len(linear_caches)} caches"
                )
            record_index = 0
            for cache_item in cache:
                if self._is_trimmable(cache_item):
                    cache_item.trim(n_rejected)
                    continue
                module, qkv, a, b, conv_state, ssm_state, mask, lengths = records[
                    record_index
                ]
                kept_mask = mask[:, :keep] if mask is not None else None
                kept_lengths = lengths
                if kept_lengths is not None:
                    import mlx.core as mx

                    kept_lengths = mx.minimum(kept_lengths, keep)
                _, new_conv_state, new_ssm_state = original_process(
                    module,
                    qkv[:, :keep],
                    a[:, :keep],
                    b[:, :keep],
                    conv_state,
                    ssm_state,
                    kept_mask,
                    kept_lengths,
                )
                cache_item[0] = new_conv_state
                cache_item[1] = new_ssm_state
                record_index += 1
            if capture is None:
                self._stash = None

        def slice_capture(self: Any, capture: Any, row_index: int):
            del self
            records, unused = capture
            row_records = []
            for module, qkv, a, b, conv_state, ssm_state, mask, lengths in records:
                row_records.append(
                    (
                        module,
                        qkv[row_index : row_index + 1],
                        a[row_index : row_index + 1],
                        b[row_index : row_index + 1],
                        conv_state[row_index : row_index + 1],
                        ssm_state[row_index : row_index + 1],
                        mask[row_index : row_index + 1] if mask is not None else None,
                        (
                            lengths[row_index : row_index + 1]
                            if lengths is not None
                            else None
                        ),
                    )
                )
            return row_records, unused

        target._capture_linear = MethodType(capture_linear, target)
        target._slice_hybrid_capture = MethodType(slice_capture, target)
        target.rollback = MethodType(rollback, target)
        logger.info("Installed native Qwen3.5/3.6 dSpark rollback adapter")
        return True


_ADAPTERS: tuple[TargetTapAdapter, ...] = (Qwen35TapAdapter(),)


def adapt_target(target: Any) -> TargetTapAdapter:
    """Install the registered adapter for an already-loaded target wrapper."""
    for adapter in _ADAPTERS:
        if adapter.matches(target):
            adapter.install(target)
            return adapter
    raise ValueError(
        f"no TargetTapAdapter registered for model_type={_model_type(target)!r}"
    )
