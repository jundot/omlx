"""Native dSpark drafter registry.

Only checkpoint-specific loading lives here.  The target model is supplied by
oMLX and is never loaded or wrapped a second time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .compat import DSparkProbe, probe_drafter
from .prequant import (
    checkpoint_quantization,
    is_prepared_checkpoint,
    load_prequantized_drafter,
)


@dataclass(frozen=True)
class DSparkLoadOptions:
    drafter: Path
    target_model: Any
    tokenizer: Any
    max_draft_tokens: str | int | None
    markov_mode: str


@dataclass(frozen=True)
class LoadedDrafter:
    model: Any
    config: Any
    format: str
    quantization: dict[str, Any]


@runtime_checkable
class DrafterHandler(Protocol):
    format_name: str

    def probe(self, path: str | Path) -> DSparkProbe: ...
    def validate_pair(self, target: Any, probe: DSparkProbe) -> None: ...
    def load(self, options: DSparkLoadOptions) -> LoadedDrafter: ...
    def memory_estimate(self, probe: DSparkProbe) -> int: ...


class _NativeHandler:
    format_name = ""

    def probe(self, path: str | Path) -> DSparkProbe:
        return probe_drafter(path, self.format_name)

    def validate_pair(self, target: Any, probe: DSparkProbe) -> None:
        del target
        if probe.format != self.format_name:
            raise ValueError(
                f"handler {self.format_name} cannot load {probe.format} checkpoint"
            )

    def memory_estimate(self, probe: DSparkProbe) -> int:
        quant = probe.quantization or {}
        bits = int(quant.get("bits", 16) or 16)
        return max(1, int(probe.weight_bytes * min(bits, 16) / 16))


class DeepSpecHandler(_NativeHandler):
    format_name = "deepspec"

    def load(self, options: DSparkLoadOptions) -> LoadedDrafter:
        from .native_load import load_drafter

        if is_prepared_checkpoint(options.drafter):
            drafter, config = load_prequantized_drafter(options.drafter)
            probe = self.probe(options.drafter)
            quantization = dict(probe.quantization or {})
        else:
            # Native request loading must never rewrite BF16 tensors.
            drafter, config = load_drafter(
                str(options.drafter), quantize=False, strict=True
            )
            quantization = {"bits": 16, "group_size": None, "status": "source"}
        self._set_markov(drafter, options.markov_mode)
        return LoadedDrafter(drafter, config, self.format_name, quantization)

    @staticmethod
    def _set_markov(drafter: Any, mode: str) -> None:
        if mode == "disabled":
            disable = getattr(drafter, "disable_markov", None)
            if callable(disable):
                disable()
        elif mode == "enabled":
            enable = getattr(drafter, "enable_markov", None)
            if not callable(enable):
                raise RuntimeError("checkpoint has no supported Markov proposal head")
            enable()


class SpeculatorsHybridHandler(_NativeHandler):
    format_name = "speculators"

    def load(self, options: DSparkLoadOptions) -> LoadedDrafter:
        from .native_load import load_dflash

        prepared = is_prepared_checkpoint(options.drafter)
        quantization = checkpoint_quantization(options.drafter) if prepared else None
        target = getattr(options.target_model, "language_model", options.target_model)
        target_args = getattr(target, "args", None)
        target_hidden_size = getattr(target_args, "hidden_size", None)
        drafter, config = load_dflash(
            str(options.drafter),
            quantize=False,
            prequantized=quantization,
            target_hidden_size=target_hidden_size,
        )
        drafter.bind(options.target_model)
        markov = getattr(drafter, "markov_head", None)
        if options.markov_mode == "disabled":
            drafter.markov_enabled = False
        elif options.markov_mode == "enabled" and markov is None:
            raise RuntimeError(
                "Markov was requested but the checkpoint has no Markov head"
            )
        else:
            drafter.markov_enabled = markov is not None
        probe = self.probe(options.drafter)
        return LoadedDrafter(
            drafter,
            config,
            self.format_name,
            dict(probe.quantization or {"bits": 16, "status": "source"}),
        )


class HiggsSidecarHandler(SpeculatorsHybridHandler):
    format_name = "higgs_sidecar"


_HANDLERS: dict[str, DrafterHandler] = {
    "deepspec": DeepSpecHandler(),
    "speculators": SpeculatorsHybridHandler(),
    "higgs_sidecar": HiggsSidecarHandler(),
}


def get_handler(format_name: str) -> DrafterHandler:
    try:
        return _HANDLERS[format_name]
    except KeyError as exc:
        raise ValueError(f"unsupported dSpark format: {format_name}") from exc


def resolve_handler(
    path: str | Path, requested_format: str = "auto"
) -> tuple[DrafterHandler, DSparkProbe]:
    probe = probe_drafter(path, requested_format)
    return get_handler(probe.format), probe
