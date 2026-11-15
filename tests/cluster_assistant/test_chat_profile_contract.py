# SPDX-License-Identifier: Apache-2.0
"""Contract: the chat-profile guard against the *real* catalogue shape.

The assistant's fakes once encoded sampling at the top level of a catalogue
entry, while the real admin catalogue (``omlx/admin/routes.py`` ``list_models``)
nests it under ``entry["settings"] = asdict(ModelSettings)``. The guard read the
wrong shape and overwrote user sampling on every start (review finding B1).

These tests build the entry exactly the way ``list_models`` does — from the
real ``ModelSettings`` dataclass, not a hand-written fake — so a drift between
the guard and the real shape fails here instead of silently returning.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

from assistant_harness import MODEL_PATH

from omlx.cluster.assistant.start import (
    CHAT_PROFILE,
    WARNING_CHAT_PROFILE_APPLIED,
    CoordinatorOps,
    ensure_chat_profile,
)
from omlx.model_settings import ModelSettings


def catalogue_entry(settings: ModelSettings) -> dict[str, Any]:
    """One entry shaped exactly like ``list_models`` builds it.

    ``_model_options`` adds no sampling keys at the top level; the only place
    sampling appears is ``settings`` (``asdict`` of the stored settings, or the
    dataclass defaults when the user never set anything).
    """

    return {
        "id": "qwen",
        "model_path": MODEL_PATH,
        "settings": asdict(settings),
    }


def profile_ops(
    listing: dict[str, Any], writes: list[tuple[str, dict[str, Any]]]
) -> CoordinatorOps:
    """Only the two catalogue calls exist; anything else is a contract breach."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the chat profile step must not prepare/stage/launch")

    async def forbidden_async(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the chat profile step must not health-check/activate")

    def write(model_id: str, patch: Any) -> dict[str, Any]:
        writes.append((model_id, dict(patch)))
        return {"success": True, "model_id": model_id}

    return CoordinatorOps(
        local=False,
        target="172.16.1.11",
        prepare=forbidden,
        stage=forbidden,
        staging_status=forbidden,
        health=forbidden_async,
        activate=forbidden_async,
        models=lambda: listing,
        write_model_settings=write,
    )


def test_every_profile_key_is_a_real_model_settings_field() -> None:
    """The profile may only write fields the settings dataclass actually has."""

    fields = asdict(ModelSettings())
    assert set(CHAT_PROFILE) <= set(fields)
    # The unset state the guard must recognise: sampling None, force off.
    assert fields["temperature"] is None
    assert fields["top_p"] is None
    assert fields["top_k"] is None
    assert fields["max_tokens"] is None
    assert fields["force_sampling"] is False


def test_a_user_temperature_in_the_real_shape_is_never_overwritten() -> None:
    settings = ModelSettings(temperature=0.2)
    writes: list[tuple[str, dict[str, Any]]] = []
    ops = profile_ops({"models": [catalogue_entry(settings)]}, writes)

    warning = asyncio.run(ensure_chat_profile(ops, MODEL_PATH))

    assert warning is None
    assert writes == []


def test_unset_settings_in_the_real_shape_get_the_full_profile() -> None:
    writes: list[tuple[str, dict[str, Any]]] = []
    ops = profile_ops({"models": [catalogue_entry(ModelSettings())]}, writes)

    warning = asyncio.run(ensure_chat_profile(ops, MODEL_PATH))

    assert writes == [("qwen", dict(CHAT_PROFILE))]
    assert warning is not None
    assert warning.code == WARNING_CHAT_PROFILE_APPLIED


def test_force_sampling_alone_is_a_user_choice_but_its_default_is_not() -> None:
    """``force_sampling`` defaults to False; only a truthy value is a choice.

    An ``is not None`` check on it would see the default False on every model
    and never apply the profile at all.
    """

    chosen: list[tuple[str, dict[str, Any]]] = []
    ops = profile_ops(
        {"models": [catalogue_entry(ModelSettings(force_sampling=True))]}, chosen
    )
    assert asyncio.run(ensure_chat_profile(ops, MODEL_PATH)) is None
    assert chosen == []

    unset: list[tuple[str, dict[str, Any]]] = []
    ops = profile_ops({"models": [catalogue_entry(ModelSettings())]}, unset)
    assert asyncio.run(ensure_chat_profile(ops, MODEL_PATH)) is not None
    assert unset == [("qwen", dict(CHAT_PROFILE))]


def test_one_user_set_key_keeps_the_whole_profile_untouched() -> None:
    """All-or-nothing: the profile is applied only when the model has none.

    A single user-set value (here ``max_tokens``) must stop the whole write,
    not just that one key — the settings PUT merges field by field, so a
    partial write would still change the user's model.
    """

    writes: list[tuple[str, dict[str, Any]]] = []
    ops = profile_ops(
        {"models": [catalogue_entry(ModelSettings(max_tokens=512))]}, writes
    )

    assert asyncio.run(ensure_chat_profile(ops, MODEL_PATH)) is None
    assert writes == []
