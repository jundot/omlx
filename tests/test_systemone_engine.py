# SPDX-License-Identifier: Apache-2.0
"""Tests for structured System One reads (P0: the engine primitive).

Two layers live here on purpose:

* pure canvas/answer math, which needs nothing; and
* the forward pass, exercised against a **tiny random-initialized**
  ``diffusion_gemma`` Model built from config — no checkpoint, no server, no
  weights on disk. The numbers are meaningless; the invariants are not.

The invariants that matter are the ones that fail silently: probabilities must
come from raw (softcapped, temperature-1) logits, noise must appear only at
slot positions, reads must not mutate the encoder cache, and a given seed must
reproduce a given answer.
"""

from __future__ import annotations

import asyncio
import importlib.util
import math
import random
import threading
import unittest

import mlx.core as mx

from omlx.engine.vlm import VLMBatchedEngine
from omlx.exceptions import InvalidRequestError
from omlx.systemone import ReadGroup, ReadSlot, SlotDistribution, normalized


def tiny_config_dict(canvas_length: int = 16) -> dict:
    """A diffusion_gemma config small enough to run on any Mac, untrained.

    ``vocab_size`` is 128 rather than 64 so the fixture can keep the real
    turn-close id (106) in its canvas templates and still be an in-range index
    into the embedding table.
    """

    return {
        "model_type": "diffusion_gemma",
        "canvas_length": canvas_length,
        "image_token_id": 60,
        "text_config": {
            "model_type": "diffusion_gemma_text",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 24,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 4,
            "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"],
            "num_experts": 4,
            "top_k_experts": 2,
            "use_bidirectional_attention": None,
            "final_logit_softcapping": 30.0,
        },
        "vision_config": None,
        "generation_config": {
            "max_denoising_steps": 1,
            "sampler_config": {
                "_cls_name": "EntropyBoundSamplerConfig",
                "entropy_bound": 0.1,
            },
            "linear_temperature_schedule_config": {
                "_cls_name": "LinearTemperatureScheduleConfig",
                "t_min": 0.4,
                "t_max": 0.8,
            },
        },
    }


def build_tiny_model(canvas_length: int = 16):
    from mlx_vlm.models.diffusion_gemma import Model, ModelConfig

    from omlx.utils.model_loading import materialize_lazy_state

    mx.random.seed(0)
    model = Model(ModelConfig.from_dict(tiny_config_dict(canvas_length)))
    # mlx binds lazily-evaluated arrays to the stream of the thread that
    # produced them. Production builds the model on the MLX executor thread and
    # then calls this same helper; a test that builds it on the main thread must
    # materialize the tree too, or the executor thread fails with
    # "There is no Stream(gpu, 0) in current thread" (#1304).
    materialize_lazy_state(model)
    return model


def build_engine(model) -> VLMBatchedEngine:
    """An engine shell with only what the read path touches.

    ``VLMBatchedEngine.__new__`` is the pattern this repo already uses for
    engine tests (see ``tests/e2e_vision_cache.py``) — the real ``__init__``
    loads a model, which is precisely what this suite must not do.
    """

    engine = VLMBatchedEngine.__new__(VLMBatchedEngine)
    engine._model_name = "tiny-diffusion-gemma"
    engine._vlm_model = model
    engine._processor = None
    engine._tokenizer = None
    engine._vlm_mtp_drafter = None
    engine._diffusion_family = "block" if model is not None else None
    engine._diffusion_lock = asyncio.Lock()
    engine._diffusion_active_requests = 0
    engine._diffusion_cancel_events = set()
    engine._enable_thinking = False
    return engine


def make_group(keys_and_labels: list[tuple[str, tuple[str, ...]]], *, seed: int = 7):
    """Build a template with one slot per question, mirroring the real shape.

    Layout is ``q1: _ q2: _ ... <turn-close>`` — labels sit after a
    single-space token, exactly as the real tokenizer resolves them.
    """

    template: list[int] = []
    slots: list[ReadSlot] = []
    for i, (key, labels) in enumerate(keys_and_labels):
        template.extend([10 + i, 8])  # "qN" + space
        position = len(template)
        template.append(0)
        label_ids = tuple(20 + 3 * i + j for j in range(len(labels)))
        slots.append(
            ReadSlot(
                key=key,
                position=position,
                label_ids=label_ids,
                label_keys=tuple(labels),
            )
        )
    template.append(106)  # turn close
    return ReadGroup(template_ids=tuple(template), slots=tuple(slots))


PROMPT_IDS = list(range(10, 40))
VOCAB = 128


class TestCanvasMath(unittest.TestCase):
    """Pure canvas construction — no model, no mlx."""

    def test_noise_appears_only_at_slot_positions(self) -> None:
        group = make_group([("a", ("yes", "no")), ("b", ("x", "y", "z"))])
        clean = group.canvas(None, 64)
        self.assertEqual(clean, list(group.template_ids))

        slot_positions = {s.position for s in group.slots}
        changed = 0
        for seed in range(20):
            noisy = group.canvas(random.Random(seed), 64)
            self.assertEqual(len(noisy), len(clean))
            for i, (before, after) in enumerate(zip(clean, noisy)):
                if i not in slot_positions:
                    self.assertEqual(
                        before, after, f"position {i} changed but is not a slot"
                    )
                elif before != after:
                    changed += 1
        self.assertGreater(changed, 0, "no draw ever touched a slot position")

    def test_canvas_is_exact_width_with_no_padding(self) -> None:
        group = make_group([("a", ("yes", "no"))])
        canvas = group.canvas(random.Random(1), 64)
        self.assertEqual(len(canvas), group.width)
        self.assertEqual(canvas[-1], 106)

    def test_noise_stays_inside_the_vocabulary(self) -> None:
        group = make_group([("a", ("yes", "no")), ("b", ("x", "y"))])
        slots = {s.position for s in group.slots}
        for seed in range(50):
            for i, token in enumerate(group.canvas(random.Random(seed), 64)):
                if i in slots:
                    self.assertIn(token, range(64))

    def test_slot_position_validation(self) -> None:
        with self.assertRaises(ValueError):
            ReadGroup(template_ids=(), slots=())
        with self.assertRaises(ValueError):
            ReadSlot(key="a", position=0, label_ids=())
        with self.assertRaises(ValueError):
            ReadGroup(
                template_ids=(1, 2, 3),
                slots=(ReadSlot(key="a", position=9, label_ids=(1,)),),
            )
        with self.assertRaises(ValueError):
            dup = ReadSlot(key="b", position=1, label_ids=(1,))
            ReadGroup(
                template_ids=(1, 2, 3),
                slots=(ReadSlot(key="a", position=1, label_ids=(2,)), dup),
            )


class TestAnswerMath(unittest.TestCase):
    def test_uniform_is_zero_confidence_and_onehot_is_one(self) -> None:
        uniform = SlotDistribution(
            key="a",
            label_ids=(1, 2, 4, 8),
            label_keys=("w", "x", "y", "z"),
            probabilities=(0.25, 0.25, 0.25, 0.25),
        )
        self.assertAlmostEqual(uniform.entropy, math.log(4), places=6)
        self.assertAlmostEqual(uniform.confidence, 0.0, places=6)

        skewed = SlotDistribution(
            key="a",
            label_ids=(1, 2),
            label_keys=("x", "y"),
            probabilities=(0.9, 0.1),
        )
        self.assertGreater(skewed.confidence, 0.0)
        self.assertLess(skewed.confidence, 1.0)
        self.assertEqual(skewed.argmax_label, "x")

        sharp = SlotDistribution(
            key="a",
            label_ids=(1, 2),
            label_keys=("x", "y"),
            probabilities=(1.0, 0.0),
        )
        self.assertAlmostEqual(sharp.entropy, 0.0, places=9)
        self.assertAlmostEqual(sharp.confidence, 1.0, places=9)
        self.assertEqual(sharp.argmax_label, "x")
        self.assertEqual(sharp.argmax_id, 1)

    def test_normalized_handles_degenerate_mass(self) -> None:
        self.assertEqual(normalized([0.0, 0.0]), (0.5, 0.5))
        self.assertEqual(normalized([1.0, 3.0]), (0.25, 0.75))


@unittest.skipUnless(
    importlib.util.find_spec("mlx_vlm.models.diffusion_gemma") is not None,
    "mlx_vlm diffusion_gemma unavailable",
)
class TestStructuredReadTinyModel(unittest.IsolatedAsyncioTestCase):
    """The forward pass, on a tiny untrained model."""

    def setUp(self) -> None:
        self.model = build_tiny_model()
        self.engine = build_engine(self.model)

    async def test_distributions_are_normalized_at_requested_positions(
        self,
    ) -> None:
        group = make_group([("a", ("yes", "no")), ("b", ("x", "y", "z"))])
        result = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=3, seed=99
        )

        self.assertEqual(len(result.distributions), 2)
        for dist, slot in zip(result.distributions, group.slots):
            self.assertEqual(dist.key, slot.key)
            self.assertEqual(dist.label_ids, slot.label_ids)
            self.assertEqual(len(dist.probabilities), slot.n_labels)
            self.assertAlmostEqual(sum(dist.probabilities), 1.0, places=5)
            self.assertTrue(all(p >= 0.0 for p in dist.probabilities))
            self.assertGreaterEqual(dist.confidence, 0.0)
            self.assertLessEqual(dist.confidence, 1.0)

        self.assertEqual(result.samples, 3)
        self.assertEqual(result.forwards, 3)
        self.assertEqual(result.canvas_tokens, 3 * group.width)
        self.assertEqual(result.prompt_tokens, len(PROMPT_IDS))
        self.assertGreater(result.prefill_ms, 0.0)
        self.assertGreater(result.decode_ms, 0.0)

    async def test_same_seed_reproduces_the_numbers(self) -> None:
        group = make_group([("a", ("yes", "no")), ("b", ("x", "y", "z"))])
        first = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=2, seed=4242
        )
        second = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=2, seed=4242
        )
        for a, b in zip(first.distributions, second.distributions):
            self.assertEqual(a.probabilities, b.probabilities)

        other = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=2, seed=11
        )
        # A different noise draw must move the numbers; identical output across
        # seeds would mean the canvas is not reaching the model.
        self.assertTrue(
            any(
                abs(x - y) > 1e-6
                for a, b in zip(first.distributions, other.distributions)
                for x, y in zip(a.probabilities, b.probabilities)
            )
        )

    async def test_reported_probabilities_are_softmax_of_raw_logits(
        self,
    ) -> None:
        """Temperature discipline: the read reports raw, not schedule-tempered."""

        group = make_group([("a", ("yes", "no"))])
        result = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=1, seed=5150
        )
        reported = result.distributions[0].probabilities

        canvas = mx.array([group.canvas(random.Random(5150), VOCAB)])
        cache = self.model.make_cache()
        cache = self.model.diffusion_prefill_cache(
            mx.array([PROMPT_IDS]),
            attention_mask=None,
            cache=cache,
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=None,
            chunk_prefill=False,
        )
        masks = self.model.diffusion_decoder_masks(canvas, cache, None)
        logits = self.model.diffusion_decoder_logits(
            canvas, cache=cache, decoder_attention_mask=masks
        )
        mx.eval(logits)
        slot = group.slots[0]
        raw = mx.take(
            logits[0][slot.position], mx.array(list(slot.label_ids), mx.uint32), 0
        )
        expected = mx.softmax(raw.astype(mx.float32), axis=-1, precise=True).tolist()

        self.assertEqual(len(expected), len(reported))
        for want, got in zip(expected, reported):
            self.assertAlmostEqual(want, got, places=6)

        # And the tempering the read must *not* apply is real and material:
        # dividing by the schedule's temperature changes the answer.
        tempered = mx.softmax(raw / 0.4, axis=-1, precise=True).tolist()
        self.assertTrue(any(abs(a - b) > 1e-4 for a, b in zip(expected, tempered)))

    async def test_sample_average_matches_independent_forwards(self) -> None:
        """Every draw sees the same cached prefix, never the previous draw.

        Reproduces the engine's rng sequence and recomputes each draw against a
        cache this test prefilled once and never commits to. If the engine
        committed a canvas to the cache — or let one draw contaminate the next —
        the two averages part company.
        """

        group = make_group([("a", ("yes", "no")), ("b", ("x", "y", "z"))])
        samples = 4
        seed = 606
        result = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=samples, seed=seed
        )
        self.assertEqual(result.forwards, samples)

        cache = self.model.make_cache()
        cache = self.model.diffusion_prefill_cache(
            mx.array([PROMPT_IDS]),
            attention_mask=None,
            cache=cache,
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=None,
            chunk_prefill=False,
        )
        rng = random.Random(seed)
        sums = {slot.key: [0.0] * slot.n_labels for slot in group.slots}
        for _ in range(samples):
            canvas = mx.array([group.canvas(rng, VOCAB)])
            masks = self.model.diffusion_decoder_masks(canvas, cache, None)
            logits = self.model.diffusion_decoder_logits(
                canvas, cache=cache, decoder_attention_mask=masks
            )
            mx.eval(logits)
            for slot in group.slots:
                raw = mx.take(
                    logits[0][slot.position],
                    mx.array(list(slot.label_ids), mx.uint32),
                    0,
                )
                draws = mx.softmax(
                    raw.astype(mx.float32), axis=-1, precise=True
                ).tolist()
                for i, value in enumerate(draws):
                    sums[slot.key][i] += value

        for dist in result.distributions:
            mean = [value / samples for value in sums[dist.key]]
            self.assertEqual(len(mean), len(dist.probabilities))
            for want, got in zip(mean, dist.probabilities):
                self.assertAlmostEqual(want, got, places=6)

    async def test_pre_cancelled_read_reports_neutral_priors(self) -> None:
        """A read that never ran still answers every key, with no opinion."""

        group = make_group([("a", ("yes", "no")), ("b", ("x", "y", "z"))])
        cancel = threading.Event()
        cancel.set()
        result = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=4, cancel_event=cancel
        )
        self.assertEqual(result.forwards, 0)
        self.assertEqual({d.key for d in result.distributions}, {"a", "b"})
        for dist in result.distributions:
            self.assertAlmostEqual(sum(dist.probabilities), 1.0, places=6)
            self.assertAlmostEqual(dist.confidence, 0.0, places=6)

    async def test_multiple_groups_share_one_prefill(self) -> None:
        g1 = make_group([("a", ("yes", "no"))])
        g2 = make_group([("b", ("x", "y")), ("c", ("p", "q", "r"))])
        result = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[g1, g2], samples=2, seed=77
        )
        self.assertEqual({d.key for d in result.distributions}, {"a", "b", "c"})
        self.assertEqual(result.forwards, 4)  # 2 groups x 2 samples, one prefill
        for dist in result.distributions:
            self.assertAlmostEqual(sum(dist.probabilities), 1.0, places=5)

    async def test_reads_do_not_mutate_the_encoder_cache(self) -> None:
        """Cache purity is the foundation: one prefill must serve N canvases."""

        cache = self.model.make_cache()
        cache = self.model.diffusion_prefill_cache(
            mx.array([PROMPT_IDS]),
            attention_mask=None,
            cache=cache,
            pixel_values=None,
            mm_token_type_ids=None,
            prefill_step_size=None,
            chunk_prefill=False,
        )
        canvas = mx.array([list(range(1, 13))])
        masks = self.model.diffusion_decoder_masks(canvas, cache, None)

        before = self.model.diffusion_decoder_logits(
            canvas, cache=cache, decoder_attention_mask=masks
        )
        mx.eval(before)
        reference = before.tolist()

        # Interleave other canvases, including a different width, the way a
        # busy lane would.
        for width in (12, 12, 9):
            other = mx.array([list(range(1, width + 1))])
            other_masks = self.model.diffusion_decoder_masks(other, cache, None)
            out = self.model.diffusion_decoder_logits(
                other, cache=cache, decoder_attention_mask=other_masks
            )
            mx.eval(out)

        after = self.model.diffusion_decoder_logits(
            canvas, cache=cache, decoder_attention_mask=masks
        )
        mx.eval(after)
        self.assertEqual(after.tolist(), reference)

    async def test_rejects_multi_step_before_touching_the_lane(self) -> None:
        group = make_group([("a", ("yes", "no"))])
        with self.assertRaises(InvalidRequestError):
            await self.engine.structured_read(
                prompt_ids=PROMPT_IDS, groups=[group], steps=4
            )

    async def test_rejects_canvas_wider_than_the_model_serves(self) -> None:
        group = make_group([("a", ("yes", "no"))])  # width 4
        engine = build_engine(build_tiny_model(canvas_length=3))
        self.assertEqual(group.width, 4)
        with self.assertRaises(InvalidRequestError) as ctx:
            await engine.structured_read(prompt_ids=PROMPT_IDS, groups=[group])
        self.assertIn("canvas", str(ctx.exception).lower())

    async def test_rejects_bad_sample_counts_and_empty_inputs(self) -> None:
        group = make_group([("a", ("yes", "no"))])
        with self.assertRaises(InvalidRequestError):
            await self.engine.structured_read(
                prompt_ids=PROMPT_IDS, groups=[group], samples=0
            )
        with self.assertRaises(InvalidRequestError):
            await self.engine.structured_read(
                prompt_ids=PROMPT_IDS, groups=[group], samples=33
            )
        with self.assertRaises(InvalidRequestError):
            await self.engine.structured_read(prompt_ids=[], groups=[group])
        with self.assertRaises(InvalidRequestError):
            await self.engine.structured_read(prompt_ids=PROMPT_IDS, groups=[])

    async def test_rejects_non_diffusion_lane(self) -> None:
        engine = build_engine(self.model)
        engine._diffusion_family = None
        with self.assertRaises(InvalidRequestError):
            await engine.structured_read(
                prompt_ids=PROMPT_IDS, groups=[make_group([("a", ("yes", "no"))])]
            )

    async def test_cancelled_read_does_not_wedge_the_lane(self) -> None:
        group = make_group([("a", ("yes", "no"))])
        await self.engine.structured_read(prompt_ids=PROMPT_IDS, groups=[group])
        self.assertEqual(self.engine._diffusion_active_requests, 0)
        self.assertFalse(self.engine._diffusion_cancel_events)
        self.assertFalse(self.engine._diffusion_lock.locked())

        task = asyncio.create_task(
            self.engine.structured_read(
                prompt_ids=PROMPT_IDS, groups=[group], samples=1, seed=3
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        # The lane is free again, and still answers.
        self.assertFalse(self.engine._diffusion_lock.locked())
        self.assertEqual(self.engine._diffusion_active_requests, 0)
        self.assertFalse(self.engine._diffusion_cancel_events)
        again = await self.engine.structured_read(
            prompt_ids=PROMPT_IDS, groups=[group], samples=1, seed=3
        )
        self.assertAlmostEqual(sum(again.distributions[0].probabilities), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
