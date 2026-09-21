# SPDX-License-Identifier: Apache-2.0
"""Tests for POST /v1/systemone (P2: the wire contract).

The engine is faked and the tokenizer is real wherever it can be. That split is
deliberate: the forward pass is P0's job and is already covered against a tiny
random-init model, while everything this layer can get silently wrong lives in
the schema — which labels survive, where the slots land, whether the scaffold got
appended twice. A fake tokenizer would pass those tests for the wrong reason.

The invariants here are the ones that fail quietly: answers keyed as the client
keyed them, ``confidence`` recomputed from the distribution the engine returned,
the pool lease released on every path including the ones that raise, and 422
detail lists that name the field the client actually got wrong.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.api import systemone_routes
from omlx.api.systemone_models import SystemOneRequest
from omlx.exceptions import (
    InvalidRequestError,
    ModelNotFoundError,
    ModelTooLargeError,
)
from omlx.systemone import (
    ReadGroup,
    SchemaError,
    SlotDistribution,
    StructuredReadResult,
    build_schema,
    confidence,
    softmax,
)

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")


def _snapshot(quant: str) -> str | None:
    hits = sorted(
        glob.glob(
            f"{HF_HUB}/models--mlx-community--diffusiongemma-26B-A4B-it-{quant}/snapshots/*"
        )
    )
    if not hits:
        return None
    has_tok = any(os.path.exists(os.path.join(s, "tokenizer.json")) for s in hits)
    return hits[-1] if has_tok else None


def _real_tokenizer(quant: str = "4bit"):
    snap = _snapshot(quant)
    if snap is None:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snap)


CANONICAL_QUESTIONS = {
    "urgent": {
        "type": "noul",
        "instructions": "Does this need a human right now?",
        "criteria": {"true": "someone must act today", "false": "it can wait"},
    },
    "team": {
        "type": "choice",
        "instructions": "Who owns it?",
        "criteria": {"outage": "platform is down", "billing": "invoice"},
    },
    "tone": {
        "type": "score",
        "instructions": "How upset?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}

STATE = "Everything is down and we have a demo at noon."


class FakeEngine:
    """Answers from the groups it was handed, with fixed probabilities.

    Probabilities come from softmax over a deterministic per-slot logit pattern,
    so the expected answer is recomputable in the test rather than hard-coded —
    a route that re-softmaxed, or tempered, the numbers would then disagree.
    """

    is_diffusion_model = True
    diffusion_canvas_length = 256

    def __init__(
        self,
        tokenizer,
        *,
        busy: bool = False,
        abort: str | None = None,
        canvas_length: int = 256,
        raise_read: Exception | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.diffusion_canvas_length = canvas_length
        self._busy = busy
        self.abort = abort
        self.raise_read = raise_read
        self.reads: list[dict] = []
        self.has_active_calls = 0

    def has_active_requests(self) -> bool:
        return self._busy

    def get_abort_requested_reason(self, _model_id):
        return self.abort

    async def structured_read(self, **kwargs):
        self.reads.append(kwargs)
        self.has_active_calls += 1
        if self.raise_read is not None:
            raise self.raise_read
        prompt_ids = kwargs["prompt_ids"]
        groups: list[ReadGroup] = kwargs["groups"]
        samples = int(kwargs.get("samples") or 1)
        seed = kwargs.get("seed")
        seed = 1234 if seed is None else int(seed)

        dists: list[SlotDistribution] = []
        for group in groups:
            for si, slot in enumerate(group.slots):
                logits = [(si + 1) * 0.7 - 0.35 * li for li in range(slot.n_labels)]
                probs = softmax(logits)
                dists.append(
                    SlotDistribution(
                        key=slot.key,
                        label_ids=slot.label_ids,
                        label_keys=slot.label_keys or slot.label_ids,
                        probabilities=probs,
                    )
                )
        return StructuredReadResult(
            distributions=dists,
            seed=seed,
            samples=samples,
            steps=int(kwargs.get("steps") or 1),
            prompt_tokens=len(prompt_ids),
            # Mirrors the engine's accounting at vlm.py:5145-5146 — one forward per
            # sample, each covering the whole canvas, so every slot is answered by
            # every forward. A fake that counted per slot would let the route's
            # diagnostics drift from the engine and the tests would not notice.
            forwards=samples * len(groups),
            canvas_tokens=samples * sum(g.width for g in groups),
            prefill_ms=1.0,
            decode_ms=2.0,
        )


class PlainEngine(FakeEngine):
    """An autoregressive engine: no canvas, so nothing to read."""

    is_diffusion_model = False


class FakeEntry:
    """Stand-in for ``EngineEntry``: only the fields the read path reads."""

    def __init__(self, model_id: str, diffusion: bool = True) -> None:
        self.model_id = model_id
        self.model_path = f"/models/{model_id}"
        self.config_model_type = "diffusion_gemma" if diffusion else "qwen3"


class FakePool:
    def __init__(self, engine, models=("m",)) -> None:
        self.engine = engine
        self.models = set(models)
        self.leased: list[str] = []
        self.released: list[str] = []
        self.get_error: Exception | None = None
        # Mirrors the pool's predicate: block-diffusion only. A PlainEngine pool
        # therefore has nothing readable, and the route has to refuse it without
        # ever reaching get_engine.
        diffusion = bool(getattr(engine, "is_diffusion_model", True))
        self.readable = set(self.models) if diffusion else set()
        self.entries = {m: FakeEntry(m, m in self.readable) for m in self.models}

    # --- the System One seam EnginePool exposes (engine_pool.py) ---
    def supports_systemone(self, model_id) -> bool:
        return model_id in self.readable

    def systemone_model_ids(self) -> list[str]:
        return sorted(self.readable)

    def get_entry(self, model_id):
        return self.entries.get(model_id)

    def get_loaded_model_ids(self) -> list[str]:
        return sorted(self.readable)

    async def get_engine(self, model_id, _lease=False, **_kw):
        if self.get_error is not None:
            raise self.get_error
        if model_id not in self.models:
            raise ModelNotFoundError(model_id, sorted(self.models))
        if _lease:
            self.leased.append(model_id)
        return self.engine

    async def release_engine(self, model_id) -> None:
        self.released.append(model_id)

    def get_abort_requested_reason(self, model_id):
        return getattr(self.engine, "abort", None)


class Metrics:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record_request_complete(self, **kwargs) -> None:
        self.calls.append(kwargs)


# The resolver maps a *configured* alias to the physical id, and is otherwise
# identity — which is how production behaves, since the pool keys its entries by
# physical id. Jev's own names (jev-latest, openjev-latest, …) are deliberately
# NOT mapped here: leaving them unmapped is what makes the route's built-in alias
# fallback the thing under test rather than the double.
_ALIAS_TO_PHYSICAL = {"alias-m": "m"}


def _app(pool, metrics=None, systemone_model: str = "") -> tuple[FastAPI, TestClient]:
    rec = metrics or Metrics()
    systemone_routes.set_systemone_getters(
        lambda: pool,
        resolve_model_id=lambda m: _ALIAS_TO_PHYSICAL.get(m, m) if m else m,
        get_metrics=lambda: rec,
        get_systemone_model=lambda: systemone_model,
    )
    app = FastAPI()
    app.include_router(systemone_routes.router)
    return app, TestClient(app)


def _body(questions=None, **kw) -> dict:
    # `is None`, not truthiness: an empty questions dict is the point of one of
    # these tests, and `questions or DEFAULT` would quietly answer another one.
    body = {
        "model": "m",
        "state": STATE,
        "questions": CANONICAL_QUESTIONS if questions is None else questions,
    }
    body.update(kw)
    return body


class TestAnswers(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = _real_tokenizer("4bit") or _real_tokenizer("mxfp4")
        if self.tok is None:
            self.skipTest("diffusiongemma tokenizer not cached")
        self.engine = FakeEngine(self.tok)
        self.pool = FakePool(self.engine)
        self.metrics = Metrics()
        _, self.client = _app(self.pool, self.metrics)

    def test_answers_are_keyed_as_the_client_keyed_them(self) -> None:
        r = self.client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(sorted(data["answers"]), ["team", "tone", "urgent"])
        self.assertEqual(data["model"], "m")

    def test_answer_shapes_per_type(self) -> None:
        answers = self.client.post("/v1/systemone", json=_body()).json()["answers"]
        self.assertEqual(set(answers["urgent"]), {"type", "noul"})
        self.assertEqual(
            set(answers["team"]), {"type", "choice", "probabilities", "confidence"}
        )
        self.assertEqual(
            set(answers["tone"]),
            {"type", "score", "legend", "probabilities", "confidence"},
        )

    def test_choice_probabilities_are_keyed_by_criteria_names(self) -> None:
        answers = self.client.post("/v1/systemone", json=_body()).json()["answers"]
        self.assertEqual(
            sorted(answers["team"]["probabilities"]), ["billing", "outage"]
        )
        self.assertIn(answers["team"]["choice"], ["billing", "outage"])

    def test_score_legend_echoes_criteria_in_order(self) -> None:
        answers = self.client.post("/v1/systemone", json=_body()).json()["answers"]
        self.assertEqual(
            answers["tone"]["legend"], {"0": "calm", "1": "annoyed", "2": "furious"}
        )

    def test_confidence_is_the_distributions_confidence(self) -> None:
        answers = self.client.post("/v1/systemone", json=_body()).json()["answers"]
        # The fake derives slot si's logits as (si+1)*0.7 - 0.35*label_index, and
        # the canonical request puts team at slot 1 with two labels. Recomputing
        # it here is the point: a route that re-softmaxed or tempered the numbers
        # would disagree with this by construction.
        team_p = softmax([1.4, 1.05])
        self.assertAlmostEqual(
            answers["team"]["confidence"], confidence(team_p), places=12
        )
        self.assertAlmostEqual(
            answers["urgent"]["noul"], softmax([0.7, 0.35])[0], places=12
        )
        self.assertTrue(0.0 <= answers["tone"]["confidence"] <= 1.0)
        self.assertAlmostEqual(
            sum(answers["tone"]["probabilities"].values()), 1.0, places=9
        )

    def test_score_is_the_expected_level(self) -> None:
        answers = self.client.post("/v1/systemone", json=_body()).json()["answers"]
        p = answers["tone"]["probabilities"]
        expected = sum(i * p[str(i)] for i in range(len(p)))
        self.assertAlmostEqual(answers["tone"]["score"], expected, places=12)

    def test_usage_counts_prompt_tokens_and_generates_nothing(self) -> None:
        data = self.client.post("/v1/systemone", json=_body()).json()
        self.assertEqual(data["usage"]["output_tokens"], 0)
        self.assertEqual(
            data["usage"]["input_tokens"], len(self.engine.reads[0]["prompt_ids"])
        )
        self.assertGreater(data["usage"]["input_tokens"], 0)

    def test_prompt_carries_the_thought_block_exactly_once(self) -> None:
        self.client.post("/v1/systemone", json=_body())
        text = self.tok.decode(self.engine.reads[0]["prompt_ids"])
        self.assertEqual(
            text.count("<|channel>thought"),
            1,
            "the thought block must appear exactly once whatever the template did",
        )

    def test_samples_and_seed_reach_the_engine_and_come_back_in_headers(self) -> None:
        r = self.client.post("/v1/systemone", json=_body(samples=8, seed=99))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(self.engine.reads[0]["samples"], 8)
        self.assertEqual(self.engine.reads[0]["seed"], 99)
        self.assertEqual(r.headers["x-systemone-samples"], "8")
        self.assertEqual(r.headers["x-systemone-seed"], "99")

    def test_request_id_is_echoed(self) -> None:
        r = self.client.post(
            "/v1/systemone", json=_body(), headers={"x-request-id": "abc-1"}
        )
        self.assertEqual(r.headers["x-request-id"], "abc-1")

    def test_metrics_record_prompt_tokens_and_resolved_model(self) -> None:
        # The request names an alias; the pool, the lease and the metrics all
        # speak the physical id it resolves to.
        self.client.post("/v1/systemone", json=_body(model="alias-m"))
        self.assertEqual(len(self.metrics.calls), 1)
        call = self.metrics.calls[0]
        self.assertEqual(call["completion_tokens"], 0)
        self.assertEqual(call["model_id"], "m")
        self.assertGreater(call["prompt_tokens"], 0)
        self.assertEqual(self.pool.leased, ["m"])

    def test_lease_is_released_after_a_good_read(self) -> None:
        self.client.post("/v1/systemone", json=_body())
        self.assertEqual(self.pool.leased, ["m"])
        self.assertEqual(self.pool.released, ["m"])

    def test_more_questions_than_one_canvas_holds_split_into_groups(self) -> None:
        questions = {f"k{i}": {"type": "noul"} for i in range(12)}
        r = self.client.post("/v1/systemone", json=_body(questions))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["answers"]), 12)
        self.assertEqual(len(self.engine.reads[0]["groups"]), 1, "canvas 256 holds 12")
        # A tiny canvas must split rather than overflow.
        self.engine.diffusion_canvas_length = 24
        r = self.client.post("/v1/systemone", json=_body(questions))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertGreater(len(self.engine.reads[-1]["groups"]), 1)


class TestErrors(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = _real_tokenizer("4bit") or _real_tokenizer("mxfp4")
        if self.tok is None:
            self.skipTest("diffusiongemma tokenizer not cached")
        self.engine = FakeEngine(self.tok)
        self.pool = FakePool(self.engine)
        _, self.client = _app(self.pool)

    def test_choice_with_one_option_is_a_422_naming_the_question(self) -> None:
        q = {"x": {"type": "choice", "criteria": {"only": "one"}}}
        r = self.client.post("/v1/systemone", json=_body(q))
        self.assertEqual(r.status_code, 422)
        locs = [d["loc"] for d in r.json()["detail"]]
        self.assertTrue(any("criteria" in [str(p) for p in loc] for loc in locs), locs)

    def test_unknown_question_type_is_a_422(self) -> None:
        q = {"x": {"type": "vibes", "criteria": {}}}
        r = self.client.post("/v1/systemone", json=_body(q))
        self.assertEqual(r.status_code, 422)

    def test_unknown_field_is_a_422_not_a_silently_defaulted_read(self) -> None:
        r = self.client.post("/v1/systemone", json=_body(criterial={"a": "b"}))
        self.assertEqual(r.status_code, 422)

    def test_no_questions_is_a_422(self) -> None:
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            SystemOneRequest.model_validate(_body({}))
        r = self.client.post("/v1/systemone", json=_body({}))
        self.assertEqual(r.status_code, 422)
        # And the schema layer refuses too, for a caller that bypasses validation.
        with self.assertRaises(SchemaError):
            build_schema({}, self.tok)

    def test_unserved_extensions_are_refused_not_dropped(self) -> None:
        for field, value in (
            ("images", ["data:image/png;base64,AA=="]),
            ("think", 64),
            ("sequential", True),
        ):
            r = self.client.post("/v1/systemone", json=_body(**{field: value}))
            self.assertEqual(r.status_code, 422, f"{field}: {r.text}")
            self.assertIn(field, str(r.json()["detail"]))

    def test_non_diffusion_model_is_404(self) -> None:
        pool = FakePool(PlainEngine(self.tok))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["error_type"], "model_not_compatible")
        # Refused before the pool was asked for an engine: nothing leased means
        # nothing loaded, which is the point — the old order loaded the model
        # and then said no.
        self.assertEqual(pool.leased, [])
        self.assertEqual(pool.released, [])

    def test_saturated_lane_answers_529_when_told_not_to_queue(self) -> None:
        pool = FakePool(FakeEngine(self.tok, busy=True))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body(queue=False))
        self.assertEqual(r.status_code, 529)
        self.assertEqual(r.json()["detail"]["error_type"], "lane_saturated")
        self.assertEqual(r.headers.get("retry-after"), "1")
        self.assertEqual(pool.released, ["m"])

    def test_saturated_lane_queues_by_default(self) -> None:
        pool = FakePool(FakeEngine(self.tok, busy=True))
        _, client = _app(pool)
        self.assertEqual(client.post("/v1/systemone", json=_body()).status_code, 200)

    def test_admin_unload_is_409_and_memory_pressure_507(self) -> None:
        pool = FakePool(FakeEngine(self.tok, abort="manual admin unload"))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 409)
        self.assertEqual(pool.released, ["m"])

        pool = FakePool(FakeEngine(self.tok, abort="hard memory pressure"))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 507)
        self.assertEqual(pool.released, ["m"])

    def test_engine_refusal_surfaces_as_422_with_its_field(self) -> None:
        pool = FakePool(
            FakeEngine(
                self.tok,
                raise_read=InvalidRequestError("canvas too wide", field="questions"),
            )
        )
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 422, r.text)
        self.assertIn("questions", str(r.json()["detail"]))
        self.assertEqual(pool.released, ["m"])

    def test_unknown_model_is_404_not_503(self) -> None:
        """The pool raises ModelNotFoundError, not HTTPException.

        Observed live: an `org/name` id against the pool's `org--name` ids came back
        503, which tells the client to retry something that can never succeed.
        """

        pool = FakePool(FakeEngine(self.tok), models=("m",))
        _, client = _app(pool)
        r = client.post(
            "/v1/systemone",
            json=_body(**{"model": "mlx-community/diffusiongemma-26B-A4B-it-4bit"}),
        )
        self.assertEqual(r.status_code, 404, r.text)
        self.assertEqual(r.json()["detail"]["error_type"], "model_not_found")
        self.assertEqual(
            pool.released, [], "no lease was taken, so none may be released"
        )

    def test_model_too_large_is_507(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        pool.get_error = ModelTooLargeError("m", 16_181_000_000, 8_000_000_000)
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 507)
        self.assertEqual(pool.released, [])

    def test_diagnostics_match_the_engines_forward_accounting(self) -> None:
        """forwards = samples x groups, canvas_tokens = forwards x width.

        Pinned to the engine's own accounting (vlm.py:5145-5146): a forward answers
        every slot on the canvas, so forwards is not slots x samples.
        """

        r = self.client.post("/v1/systemone", json=_body(samples=4))
        self.assertEqual(r.status_code, 200, r.text)
        groups = self.engine.reads[-1]["groups"]
        self.assertEqual(int(r.headers["x-systemone-forwards"]), 4 * len(groups))
        self.assertEqual(
            int(r.headers["x-systemone-canvas-tokens"]),
            4 * sum(g.width for g in groups),
        )

    def test_pool_failure_is_503_not_500(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        pool.get_error = RuntimeError("disk on fire")
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"]["error_type"], "backend_unavailable")

    def test_unmounted_server_answers_503(self) -> None:
        systemone_routes.set_systemone_getters(lambda: None)
        app = FastAPI()
        app.include_router(systemone_routes.router)
        r = TestClient(app).post("/v1/systemone", json=_body())
        self.assertEqual(r.status_code, 503)


class TestSeam(unittest.TestCase):
    def test_the_wire_layer_imports_no_mlx(self) -> None:
        """The route layer must not be where mlx gets pulled into a request.

        Checked in a fresh interpreter: in-process the answer depends on which
        test ran first, which is not a property worth asserting.
        """

        code = (
            "import sys; import omlx.api.systemone_routes as r;"
            " print('mlx' in sys.modules,"
            " ','.join(sorted(x.path for x in r.router.routes)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(out.returncode, 0, out.stderr[-400:])
        loaded, paths = out.stdout.split()
        # The router owns these paths unprefixed; the /jev prefix is applied by
        # server.py at include time, so a drift here means the contract moved.
        self.assertEqual(paths.split(","), ["/v1/models", "/v1/systemone"])
        self.assertEqual(
            loaded,
            "False",
            "importing the route module must not import mlx — that is the seam",
        )


class TestJevListing(unittest.TestCase):
    """``GET /jev/v1/models`` — the shape the TypeSafe SDK parses, and its contents.

    The SDK's ``ListModelsResponse`` requires the ``models`` key and
    ``ModelMetadata`` requires ``name``, ``description`` and ``release_date`` on
    every entry, so a missing key is a client-side ``ValidationError`` rather
    than a cosmetic gap.
    """

    def setUp(self) -> None:
        self.tok = _real_tokenizer("4bit") or _real_tokenizer("mxfp4")
        if self.tok is None:
            self.skipTest("diffusiongemma tokenizer not cached")

    def test_the_wrapper_is_models_not_openai_data(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        _, client = _app(pool)
        body = client.get("/v1/models").json()
        self.assertEqual(list(body), ["models"])
        self.assertTrue(body["models"])

    def test_every_entry_carries_the_three_required_keys(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        _, client = _app(pool)
        for entry in client.get("/v1/models").json()["models"]:
            self.assertEqual(sorted(entry), ["description", "name", "release_date"])
            self.assertTrue(entry["name"])
            self.assertTrue(entry["description"])
            self.assertRegex(entry["release_date"], r"^\d{4}-\d{2}-\d{2}$")

    def test_the_jev_aliases_are_listed_and_name_their_target(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        _, client = _app(pool, systemone_model="m")
        entries = {m["name"]: m for m in client.get("/v1/models").json()["models"]}
        self.assertIn("m", entries)
        for alias in systemone_routes.JEV_MODEL_ALIASES:
            self.assertIn(alias, entries)
            self.assertEqual(entries[alias]["description"], "Alias for m.")

    def test_an_autoregressive_only_pool_lists_nothing(self) -> None:
        # Naming a model this route answers 404 for would be a lie to the client.
        pool = FakePool(PlainEngine(self.tok))
        _, client = _app(pool)
        self.assertEqual(client.get("/v1/models").json()["models"], [])


class TestAliasResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.tok = _real_tokenizer("4bit") or _real_tokenizer("mxfp4")
        if self.tok is None:
            self.skipTest("diffusiongemma tokenizer not cached")

    def test_jev_alias_reaches_the_configured_checkpoint(self) -> None:
        pool = FakePool(FakeEngine(self.tok), models=("m", "second"))
        _, client = _app(pool, systemone_model="second")
        r = client.post("/v1/systemone", json=_body(model="jev-latest"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(pool.leased, ["second"])
        self.assertEqual(pool.released, ["second"])
        # Answered under the name the client asked for, leased by the physical id.
        self.assertEqual(r.json()["model"], "jev-latest")

    def test_alias_without_configuration_picks_a_readable_model(self) -> None:
        pool = FakePool(FakeEngine(self.tok), models=("m",))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body(model="openjev-latest"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(pool.leased, ["m"])

    def test_configured_model_that_is_not_readable_falls_back(self) -> None:
        pool = FakePool(FakeEngine(self.tok), models=("m", "qwen"))
        pool.readable.discard("qwen")
        _, client = _app(pool, systemone_model="qwen")
        r = client.post("/v1/systemone", json=_body(model="jev-latest"))
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(pool.leased, ["m"])

    def test_unknown_name_is_not_found_not_incompatible(self) -> None:
        pool = FakePool(FakeEngine(self.tok))
        _, client = _app(pool)
        r = client.post("/v1/systemone", json=_body(model="nope"))
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.json()["detail"]["error_type"], "model_not_found")
        self.assertEqual(pool.leased, [])


class TestPrefixPairing(unittest.TestCase):
    """Both endpoints under one prefix, because the SDK has one base_url."""

    def setUp(self) -> None:
        self.tok = _real_tokenizer("4bit") or _real_tokenizer("mxfp4")
        if self.tok is None:
            self.skipTest("diffusiongemma tokenizer not cached")

    def test_both_endpoints_resolve_under_the_same_prefix(self) -> None:
        from omlx.server import SYSTEMONE_URL_PREFIX

        pool = FakePool(FakeEngine(self.tok))
        systemone_routes.set_systemone_getters(
            lambda: pool,
            resolve_model_id=lambda m: m,
            get_metrics=lambda: Metrics(),
            get_systemone_model=lambda: "",
        )
        app = FastAPI()
        app.include_router(systemone_routes.router, prefix=SYSTEMONE_URL_PREFIX)
        client = TestClient(app)
        self.assertEqual(
            client.get(f"{SYSTEMONE_URL_PREFIX}/v1/models").status_code, 200
        )
        self.assertEqual(
            client.post(
                f"{SYSTEMONE_URL_PREFIX}/v1/systemone", json=_body()
            ).status_code,
            200,
        )


if __name__ == "__main__":
    unittest.main()
