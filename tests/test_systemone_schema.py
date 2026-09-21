# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure System One schema and numerics layers.

Tokenizer-only: no weights, no server, no mlx. The numbers asserted here were
read off OpenJev's own ``resolve_template`` running against the same cached
tokenizer (``notes/openjev-systemone-on-omlx.md`` §10), so a mismatch means the
port drifted, not that a fixture was mistyped.
"""

from __future__ import annotations

import glob
import math
import os
import random
import unittest

from omlx.systemone import read as rd
from omlx.systemone import schema as sch
from omlx.systemone.types import ReadGroup, ReadSlot

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")

#: The canonical request from OpenJev's README.
QUESTIONS = {
    "urgent": {
        "type": "noul",
        "instructions": "Does the customer need a reply within the hour?",
    },
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {
            "outage": "service down",
            "billing": "charges, refunds",
            "feature": "requests, how-to",
        },
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}

#: Verified output of OpenJev's resolve_template on the above.
EXPECTED_TEMPLATE_LEN = 19
EXPECTED_POSITIONS = [7, 12, 18]
EXPECTED_LABEL_IDS = [[11262, 951], [562, 603, 565], [236771, 236770, 236778]]


def _snapshot(quant: str) -> str | None:
    pat = (
        f"{HF_HUB}/models--mlx-community--diffusiongemma-26B-A4B-it-{quant}/snapshots/*"
    )
    hits = sorted(glob.glob(pat))
    if not hits or not os.path.exists(f"{hits[-1]}/tokenizer.json"):
        return None
    return hits[-1]


def _tokenizer(quant: str):
    from transformers import AutoTokenizer

    snap = _snapshot(quant)
    if snap is None:
        raise unittest.SkipTest(f"diffusiongemma-{quant} tokenizer not cached")
    return AutoTokenizer.from_pretrained(snap)


class FakeTokenizer:
    """A tokenizer whose labels tokenize to differing lengths."""

    def encode(self, text, *, add_special_tokens=False):  # noqa: ANN001, D102
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, **kw):  # noqa: ANN003, D102
        return [1, 2, 3]


def _noul_questions(n: int) -> dict:
    return {f"k{i}": {"type": "noul"} for i in range(n)}


def _plain_group_qs(n: int) -> list[dict]:
    return [
        {
            "key": f"q{i}",
            "id": f"q{i}",
            "type": "noul",
            "labels": ["yes", "no"],
            "choices": [("yes", ""), ("no", "")],
            "instructions": "",
        }
        for i in range(1, n + 1)
    ]


class TestSchema(unittest.TestCase):
    def test_build_schema_ids_are_positional(self) -> None:
        schema = sch.build_schema(QUESTIONS, FakeTokenizer())
        self.assertEqual([q["id"] for q in schema.questions], ["q1", "q2", "q3"])
        self.assertEqual([q["key"] for q in schema.questions], list(QUESTIONS))
        self.assertEqual(schema.fmt, "lines")

    def test_noul_labels_are_yes_no_in_that_order(self) -> None:
        schema = sch.build_schema({"a": {"type": "noul"}}, FakeTokenizer())
        self.assertEqual(schema.questions[0]["labels"], ["yes", "no"])
        self.assertEqual([c[0] for c in schema.questions[0]["choices"]], ["yes", "no"])

    def test_score_legend_echoes_criteria_exactly(self) -> None:
        schema = sch.build_schema(
            {"a": {"type": "score", "criteria": ["calm", "upset"]}},
            FakeTokenizer(),
        )
        self.assertEqual(schema.questions[0]["legend"], ["calm", "upset"])
        self.assertEqual(schema.questions[0]["labels"], ["0", "1"])

    def test_format_switches_past_ten_questions(self) -> None:
        self.assertEqual(
            sch.build_schema(_noul_questions(11), FakeTokenizer()).fmt, "indexed"
        )
        self.assertEqual(
            sch.build_schema(_noul_questions(10), FakeTokenizer()).fmt, "lines"
        )

    def test_rejects_bad_shapes(self) -> None:
        tok = FakeTokenizer()
        with self.assertRaises(sch.SchemaError):
            sch.build_schema({}, tok)
        with self.assertRaises(sch.SchemaError):
            sch.build_schema({"a": {"type": "choice", "criteria": {"x": "1"}}}, tok)
        with self.assertRaises(sch.SchemaError):
            sch.build_schema({"a": {"type": "score", "criteria": ["1"]}}, tok)
        with self.assertRaises(sch.SchemaError):
            sch.build_schema(
                {"a": {"type": "score", "criteria": [str(i) for i in range(11)]}},
                tok,
            )
        with self.assertRaises(sch.SchemaError) as ctx:
            sch.build_schema({"a": {"type": "wat"}}, tok)
        self.assertEqual(ctx.exception.loc, ["body", "questions", "a", "type"])

    def test_text_of_handles_string_object_array(self) -> None:
        self.assertEqual(sch.text_of(" hi "), "hi")
        self.assertEqual(sch.text_of(None), "")
        self.assertEqual(sch.text_of(["a", "b"]), '["a", "b"]')


class TestSystemText(unittest.TestCase):
    def test_every_label_is_spelled_out(self) -> None:
        schema = sch.build_schema(QUESTIONS, FakeTokenizer())
        text = sch.system_text(list(schema.questions), schema.fmt)
        for label in ["yes", "no", "A", "B", "C", "0", "1", "2"]:
            self.assertIn(label, text)
        self.assertIn("q1", text)
        self.assertIn(sch.FORMATS["lines"][2], text)

    def test_chunked_note_only_when_chunked(self) -> None:
        schema = sch.build_schema(QUESTIONS, FakeTokenizer())
        qs = list(schema.questions)
        self.assertNotIn("may cover only some", sch.system_text(qs, "lines"))
        self.assertIn("may cover only some", sch.system_text(qs, "lines", chunked=True))


class TestNumerics(unittest.TestCase):
    def test_softmax_matches_brute_force(self) -> None:
        logits = [1.0, 2.0, 0.5, -3.0]
        got = rd.softmax(logits)
        ex = [math.exp(x) for x in logits]
        want = [e / sum(ex) for e in ex]
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=12)
        self.assertAlmostEqual(sum(got), 1.0, places=12)

    def test_softmax_is_shift_invariant(self) -> None:
        a = rd.softmax([1.0, 2.0, 3.0])
        b = rd.softmax([1001.0, 1002.0, 1003.0])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)

    def test_tempering_changes_the_distribution(self) -> None:
        """The guard against reporting schedule-tempered logits.

        Tempering preserves argmax and destroys calibration, so a test that
        only checked argmax would pass either way. The probabilities must move.
        """
        logits = [2.0, 0.0, -1.0]
        raw = rd.softmax(logits)
        tempered = rd.softmax([x / 0.4 for x in logits])
        self.assertEqual(raw.index(max(raw)), tempered.index(max(tempered)))
        self.assertGreater(abs(raw[0] - tempered[0]), 0.05)

    def test_slot_distribution_selects_before_renormalizing(self) -> None:
        row = [0.0] * 100
        row[7], row[9] = 1.0, 3.0
        p = rd.slot_distribution(row, [7, 9])
        ex7, ex9 = math.exp(1.0), math.exp(3.0)
        self.assertAlmostEqual(p[0], ex7 / (ex7 + ex9), places=12)
        self.assertAlmostEqual(p[1], ex9 / (ex7 + ex9), places=12)
        self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_confidence_endpoints(self) -> None:
        self.assertAlmostEqual(rd.confidence([1.0, 0.0]), 1.0, places=9)
        self.assertAlmostEqual(rd.confidence([0.5, 0.5]), 0.0, places=9)
        self.assertAlmostEqual(rd.confidence([1 / 3, 1 / 3, 1 / 3]), 0.0, places=9)

    def test_entropy_of_uniform_is_ln_k(self) -> None:
        self.assertAlmostEqual(rd.entropy([0.5, 0.5]), math.log(2), places=12)

    def test_mean_probabilities(self) -> None:
        self.assertEqual(rd.mean_probabilities([[1.0, 0.0], [0.0, 1.0]]), (0.5, 0.5))
        with self.assertRaises(ValueError):
            rd.mean_probabilities([[1.0, 0.0], [1.0]])

    def test_to_answer_shapes(self) -> None:
        noul = rd.to_answer(
            {"type": "noul", "choices": [("yes", ""), ("no", "")]},
            [0.8, 0.2],
        )
        self.assertEqual(noul, {"type": "noul", "noul": 0.8})

        choice = rd.to_answer(
            {"type": "choice", "choices": [("a", ""), ("b", "")]}, [0.3, 0.7]
        )
        self.assertEqual(choice["choice"], "b")
        self.assertAlmostEqual(choice["probabilities"]["b"], 0.7, places=12)

        score = rd.to_answer(
            {
                "type": "score",
                "choices": [("0", ""), ("1", ""), ("2", "")],
                "legend": ["calm", "upset", "furious"],
            },
            [0.5, 0.25, 0.25],
        )
        self.assertAlmostEqual(score["score"], 0.75, places=12)
        self.assertEqual(score["legend"]["2"], "furious")


class TestCanvas(unittest.TestCase):
    def test_canvas_width_is_exact_not_padded(self) -> None:
        self.assertEqual(sch.canvas_width(19, 256), 20)
        with self.assertRaises(sch.SchemaError):
            sch.canvas_width(256, 256)

    def test_noise_lands_only_at_slots(self) -> None:
        slots = (
            ReadSlot(key="a", position=7, label_ids=(1, 2)),
            ReadSlot(key="b", position=12, label_ids=(3, 4)),
        )
        group = ReadGroup(template_ids=tuple(range(20)), slots=slots)
        clean = group.canvas(None, 262144)
        self.assertEqual(clean, list(range(20)))
        for seed in range(20):
            noisy = group.canvas(random.Random(seed), 262144)
            moved = {i for i, (a, b) in enumerate(zip(clean, noisy)) if a != b}
            self.assertTrue(
                moved.issubset({7, 12}), f"seed {seed} moved {sorted(moved)}"
            )


class TestReadGroups(unittest.TestCase):
    def test_groups_pack_in_order_and_split_at_capacity(self) -> None:
        tok = FakeTokenizer()
        scaffold = tuple(tok.encode(sch.SCAFFOLD_TEXT, add_special_tokens=False))
        qs = _plain_group_qs(5)
        one = sch.groups(tok, qs, "lines", scaffold=scaffold, canvas_length=0)
        self.assertEqual(len(one), 1)
        tight = sch.groups(tok, qs, "lines", scaffold=scaffold, canvas_length=30)
        self.assertGreater(len(tight), 1)
        flat = [q["id"] for g in tight for q in g]
        self.assertEqual(flat, [q["id"] for q in qs])


class TestAgainstRealTokenizer(unittest.TestCase):
    """The numbers that must not drift. Skipped if the checkpoint is absent."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _tokenizer("mxfp4")

    def test_canonical_template_and_slots(self) -> None:
        schema = sch.build_schema(QUESTIONS, self.tok)
        scaffold = sch.scaffold_ids(self.tok)
        template, slots = sch.resolve_template(
            self.tok, list(schema.questions), schema.fmt, scaffold=scaffold
        )
        self.assertEqual(len(template), EXPECTED_TEMPLATE_LEN)
        self.assertEqual([s.position for s in slots], EXPECTED_POSITIONS)
        self.assertEqual([list(s.label_ids) for s in slots], EXPECTED_LABEL_IDS)
        self.assertEqual(
            [s.label_keys for s in slots],
            [("yes", "no"), ("outage", "billing", "feature"), ("0", "1", "2")],
        )

    def test_scaffold_is_four_tokens(self) -> None:
        self.assertEqual(list(sch.scaffold_ids(self.tok)), [100, 45518, 107, 101])

    def test_choice_labels_stay_one_token(self) -> None:
        labels = sch.choice_labels(self.tok)
        self.assertGreaterEqual(len(labels), 26)
        base = [int(t) for t in self.tok.encode("q1: A", add_special_tokens=False)]
        for label in labels:
            text = f"q1: {label}"
            e = [int(t) for t in self.tok.encode(text, add_special_tokens=False)]
            self.assertEqual(len(e), len(base), f"{label!r} is not one token")

    def test_mxfp4_template_does_not_emit_the_scaffold(self) -> None:
        """The target's scaffold class, asserted rather than assumed."""
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertFalse(sch.template_emits_scaffold(self.tok, msgs))

    def test_prompt_carries_the_scaffold_exactly_once(self) -> None:
        schema = sch.build_schema(QUESTIONS, self.tok)
        sys_text = sch.system_text(list(schema.questions), schema.fmt)
        ids = sch.build_prompt_ids(self.tok, sys_text, "Everything is down.")
        scaffold = list(sch.scaffold_ids(self.tok))
        hits = sum(
            1
            for i in range(len(ids) - len(scaffold) + 1)
            if ids[i : i + len(scaffold)] == scaffold
        )
        self.assertEqual(hits, 1)

    def test_read_groups_compile(self) -> None:
        schema = sch.build_schema(QUESTIONS, self.tok)
        groups = sch.read_groups(self.tok, schema, canvas_length=256)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].width, EXPECTED_TEMPLATE_LEN)
        self.assertEqual([s.position for s in groups[0].slots], EXPECTED_POSITIONS)

    def test_labels_that_do_not_share_a_slot_are_refused(self) -> None:
        """A label needing two tokens cannot occupy one canvas position."""

        class TwoTokenLabel(FakeTokenizer):
            def encode(self, text, *, add_special_tokens=False):  # noqa: ANN001, D102
                ids = [ord(c) for c in text]
                return ids + [999] if text.endswith("no") else ids

        with self.assertRaises(sch.SchemaError):
            sch.resolve_template(
                TwoTokenLabel(), _plain_group_qs(1), "lines", scaffold=()
            )


@unittest.skipUnless(_snapshot("8bit") is not None, "diffusiongemma-8bit not cached")
class TestAgainstAffine8Bit(unittest.TestCase):
    """The other scaffold class, where the template already emits the thought
    block. This is the checkpoint that makes the unconditional append duplicate
    it, so the same exactly-once assertion has to hold here too (notes §10).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tok = _tokenizer("8bit")

    def test_template_emits_the_scaffold(self) -> None:
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        self.assertTrue(sch.template_emits_scaffold(self.tok, msgs))

    def test_prompt_carries_the_scaffold_exactly_once(self) -> None:
        ids = sch.build_prompt_ids(self.tok, "Answer about the state.", "Down again.")
        scaffold = list(sch.scaffold_ids(self.tok))
        hits = sum(
            1
            for i in range(len(ids) - len(scaffold) + 1)
            if ids[i : i + len(scaffold)] == scaffold
        )
        self.assertEqual(hits, 1)


if __name__ == "__main__":
    unittest.main()
