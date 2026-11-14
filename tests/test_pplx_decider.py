# SPDX-License-Identifier: Apache-2.0
"""
pplx-decider (Perplexity Decider v1.3) decision models.

Covers the ``joint_config.json`` detection (decision_kind / detect_model_type),
the wrapper-class selection in the decision engine, the checkpoint config
validation, and the chunked prompt protocol of the release runtime: context and
question chunks, the readout markers, the ``separate``/``isolated`` layouts and
the escaping of literal special tokens in user content.

No backbone weights are needed: the tests drive the model against a stub
backbone and a stub tokenizer.
"""

import json
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from omlx.engine.decision import _MODEL_CLASSES  # noqa: E402
from omlx.model_discovery import decision_kind, detect_model_type  # noqa: E402
from omlx.models.decision import (  # noqa: E402
    DecisionBackbone,
    DecisionContextLengthError,
    DecisionRequestError,
    has_vision_module,
)
from omlx.models.pplx_decider import (  # noqa: E402
    CONTEXT_END,
    LOCATE_CUTOFF,
    LOCATE_ADAPTER_WEIGHTS,
    LOCATE_DIR,
    LOCATE_FORMAT,
    LOCATE_HEAD_WEIGHTS,
    LOCATE_MARKER,
    normalize_question,
    PplxDeciderModel,
    _locate_sha256,
)

MARKER = "<|fim_suffix|>"
SEPARATOR = "<|fim_middle|>"
CODES = ["A", "B", "C"]
HIDDEN = 8


class StubTokenizer:
    """Character-level tokenizer with the release's special tokens."""

    def __init__(self):
        self.specials = {
            "<|im_start|>": 502,
            "<|im_end|>": 503,
            "<|vision_start|>": 504,
            "<|vision_end|>": 505,
            "<|image_pad|>": 506,
            "<|video_pad|>": 507,
            MARKER: 500,
            SEPARATOR: 501,
        }
        self.all_special_tokens = list(self.specials)
        self._reverse = {value: key for key, value in self.specials.items()}
        self._next = 900
        self._ordered = sorted(self.specials.items(), key=lambda kv: -len(kv[0]))

    def add_special_tokens(self, mapping, replace_extra_special_tokens=True):
        for token in mapping.get("extra_special_tokens", []):
            if token not in self.specials:
                self.specials[token] = self._next
                self._next += 1
                self.all_special_tokens = list(self.specials)
                self._reverse[self.specials[token]] = token
                self._ordered = sorted(self.specials.items(), key=lambda kv: -len(kv[0]))

    def get_vocab(self):
        return dict(self.specials)

    def encode(self, text, add_special_tokens=False):
        ids, index = [], 0
        while index < len(text):
            for token, token_id in self._ordered:
                if text.startswith(token, index):
                    ids.append(token_id)
                    index += len(token)
                    break
            else:
                ids.append(100 + ord(text[index]) % 400)
                index += 1
        return ids

    def convert_ids_to_tokens(self, token_id):
        if token_id in self._reverse:
            return self._reverse[token_id]
        return "chr" + str(token_id)

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    ):
        out = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    if "image" in item
                    else item["text"]
                    for item in content
                )
            out += f"<|im_start|>{message['role']}\n{content.rstrip()}<|im_end|>\n"
        return out


class StubBackbone:
    """Stand-in for DecisionBackbone: tokenizer, config and tokenization only."""

    def __init__(self, tokenizer=None, limit=262144):
        self.tokenizer = tokenizer or StubTokenizer()
        self._text_model = StubTextModel()
        self.config = {"text_config": {"hidden_size": HIDDEN}}
        self.image_processor = None
        self.max_position_embeddings = limit

    @property
    def has_vision(self):
        return False

    def load(self):
        return None

    def close(self):
        return None

    def tokenize(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


def _joint_config() -> dict:
    return {
        "version": 4,
        "template": "joint-chunked-readout-v1",
        "marker": MARKER,
        "readout_offset": 0,
        "separator": SEPARATOR,
    }


def _decision_config(**overrides) -> dict:
    config = {
        "format_version": 2,
        "model_type": "qwen3_5",
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "temperature": 1.2263882168038833,
        "codes": CODES,
        "token_ids": [100 + ord(code) % 400 for code in CODES],
        "max_pixels": 1638400,
        "default_question_mode": "separate",
    }
    config.update(overrides)
    return config


def _write_model_dir(tmp_path, config=None, joint=None, readout=True, name="checkpoint"):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    (model_dir / "decision_config.json").write_text(json.dumps(config or _decision_config()))
    (model_dir / "joint_config.json").write_text(json.dumps(joint or _joint_config()))
    if readout:
        mx.save_safetensors(
            str(model_dir / "readout.safetensors"),
            {"weight": mx.zeros((len(CODES), HIDDEN))},
        )
    return model_dir


def _loaded_model(tmp_path, **config_overrides):
    """A model whose backbone is a stub, but whose load() ran for real."""
    path = _write_model_dir(tmp_path, config=_decision_config(**config_overrides))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    model.load()
    return model


# --------------------------------------------------------------------------
# decision_kind / detect_model_type


def test_decision_kind_pplx_decider(tmp_path):
    assert decision_kind(_write_model_dir(tmp_path)) == "pplx_decider"


def test_decision_kind_needs_the_readout(tmp_path):
    assert decision_kind(_write_model_dir(tmp_path, readout=False)) is None


def test_decision_kind_ignores_a_foreign_template(tmp_path):
    joint = _joint_config() | {"template": "something-else"}
    assert decision_kind(_write_model_dir(tmp_path, joint=joint)) is None


def test_detect_model_type_pplx_decider_is_decision(tmp_path):
    assert detect_model_type(_write_model_dir(tmp_path)) == "decision"


def test_decision_classes_route_the_kind():
    from omlx.models.pplx_decider import PplxDeciderModel as Model

    assert _MODEL_CLASSES["pplx_decider"] is Model


# --------------------------------------------------------------------------
# config validation


def test_check_config_rejects_a_bad_attention_mode(tmp_path):
    path = _write_model_dir(tmp_path, config=_decision_config(attention_mode="causal"))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="attention_mode"):
        model.load()


def test_check_config_rejects_a_missing_temperature(tmp_path):
    config = _decision_config()
    config.pop("temperature")
    path = _write_model_dir(tmp_path, config=config)
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="temperature"):
        model.load()


def test_load_reads_codes_temperature_and_marker(tmp_path):
    model = _loaded_model(tmp_path)
    assert model.codes == CODES
    assert model.temperature == pytest.approx(1.2263882168038833)
    assert model.marker_id == 500
    assert model.separator_id == 501
    assert model.mode == "separate"


def test_load_rejects_a_code_that_does_not_tokenize_to_its_id(tmp_path):
    path = _write_model_dir(tmp_path, config=_decision_config(token_ids=[1, 2, 3]))
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone()
    with pytest.raises(ValueError, match="tokenize"):
        model.load()


# --------------------------------------------------------------------------
# prompt protocol


def test_state_text_and_escaping(tmp_path):
    model = _loaded_model(tmp_path)
    assert model.state_text("hello") == "State:\nhello"
    assert model.state_text("") == "State:\n(empty)"
    # A literal marker in the state must not read as a readout marker.
    escaped = model.state_text(f"a {MARKER} b")
    assert MARKER not in escaped
    assert "\uff1c|fim_suffix|\uff1e" in escaped


def test_question_text_lists_the_options_under_their_codes(tmp_path):
    model = _loaded_model(tmp_path)
    question = model._normalize("q", {"type": "choice", "instructions": "Pick", "criteria": {"one": "first", "two": None}})
    assert model.question_text(question) == (
        "Question:\nPick\nOptions:\nA: one: first\nB: two"
    )


def test_question_text_defaults_the_instructions(tmp_path):
    model = _loaded_model(tmp_path)
    question = model._normalize("q", {"type": "noul"})
    text = model.question_text(question)
    assert "Choose the best matching option." in text
    assert text.endswith("A: No / false\nB: Yes / true")


def test_context_text_keeps_the_state_and_leaves_the_turn_open(tmp_path):
    model = _loaded_model(tmp_path)
    state = "trailing spaces   "
    text = model.context_text(state, 0)
    assert text.startswith("<|im_start|>system\nClassify the shared state")
    assert text.endswith(f"<|im_start|>user\nState:\n{state}")
    assert CONTEXT_END not in text


# --------------------------------------------------------------------------
# request validation


def _request(**overrides):
    request = {
        "state": "the state",
        "questions": {"q": {"type": "noul", "instructions": "Is it true?"}},
    }
    request.update(overrides)
    return request


def test_encode_separate_gives_one_sequence_per_question(tmp_path):
    model = _loaded_model(tmp_path)
    request = _request(
        questions={
            "a": {"type": "noul"},
            "b": {"type": "choice", "criteria": {"x": None, "y": None}},
        }
    )
    plan = model.encode(request)
    assert [sequence.keys for sequence in plan.sequences] == [("a",), ("b",)]
    for sequence in plan.sequences:
        assert sequence.markers == [len(sequence.ids) - 1]
        assert sequence.ids[sequence.markers[0]] == model.marker_id
        assert sequence.ids[len(sequence.ids) - 2] != model.marker_id
    # The context (system prompt, state, separator) is shared by both sequences.
    head = list(plan.sequences[0].ids[: len(plan.sequences[1].ids)])
    assert head != list(plan.sequences[1].ids)
    assert list(plan.sequences[0].ids[:20]) == list(plan.sequences[1].ids[:20])


def test_encode_isolated_puts_every_question_in_one_sequence(tmp_path):
    model = _loaded_model(tmp_path)
    request = _request(
        mode="isolated",
        questions={"a": {"type": "noul"}, "b": {"type": "noul"}},
    )
    plan = model.encode(request)
    assert len(plan.sequences) == 1
    sequence = plan.sequences[0]
    assert sequence.keys == ("a", "b")
    assert len(sequence.markers) == 2
    assert [sequence.ids[marker] for marker in sequence.markers] == [model.marker_id] * 2


def test_encode_rejects_an_unknown_mode(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="mode"):
        model.encode(_request(mode="together"))


def test_encode_rejects_a_single_level_score(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="score"):
        model.encode(_request(questions={"q": {"type": "score", "criteria": ["only"]}}))


def test_encode_rejects_more_options_than_codes(tmp_path):
    model = _loaded_model(tmp_path)
    criteria = {f"option{index}": None for index in range(4)}
    with pytest.raises(DecisionRequestError, match="options"):
        model.encode(_request(questions={"q": {"type": "choice", "criteria": criteria}}))


def test_encode_rejects_a_question_over_the_context_limit(tmp_path):
    path = _write_model_dir(tmp_path)
    model = PplxDeciderModel(str(path))
    model.backbone = StubBackbone(limit=32)
    model.load()
    with pytest.raises(DecisionContextLengthError):
        model.encode(_request())


def _png_data_uri() -> str:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_encode_rejects_images_without_a_vision_tower(tmp_path):
    model = _loaded_model(tmp_path)
    with pytest.raises(DecisionRequestError, match="vision"):
        model.encode(_request(images=[_png_data_uri()]))


# --------------------------------------------------------------------------
# answer formatting


def test_run_reads_out_probabilities_per_question(tmp_path, monkeypatch):
    """Drive run() with a stubbed encode/backbone forward."""
    model = _loaded_model(tmp_path)
    plan = model.encode(_request(questions={"a": {"type": "noul"}}))
    sequence = plan.sequences[0]

    class StubTextModel:
        class Norm:
            @staticmethod
            def __call__(hidden):
                return hidden

        def __init__(self):
            self.layers = []
            self.norm = StubTextModel.Norm()

    model.backbone._text_model = StubTextModel()
    model.backbone.embed_tokens = lambda ids: mx.zeros((1, len(ids), HIDDEN))

    def fake_encode_markers(embeds, positions, chunk_size, markers):
        hidden = mx.zeros((len(markers), HIDDEN))
        yield len(sequence.ids)
        return hidden

    monkeypatch.setattr(model, "_encode_markers", fake_encode_markers)
    steps = model.run(plan, lambda: 2048)
    with pytest.raises(StopIteration) as stop:
        while True:
            next(steps)
    result = stop.value.value
    assert set(result["answers"]) == {"a"}
    assert result["answers"]["a"]["type"] == "noul"
    assert result["answers"]["a"]["noul"] == pytest.approx(0.5)
    assert result["input_tokens"] == len(sequence.ids)


# The model exposes the question normalizer as a method for the tests above.
def _normalize(self, question_id, question):
    from omlx.models.pplx_decider import normalize_question

    return normalize_question(question_id, question)


PplxDeciderModel._normalize = _normalize


def test_normalize_question_shapes():
    from omlx.models.pplx_decider import normalize_question

    assert normalize_question("q", {"type": "noul"}).keys == ["false", "true"]
    choice = normalize_question("q", {"type": "choice", "criteria": {"k": "d"}})
    assert choice.texts == ["k: d"]
    score = normalize_question("q", {"type": "score", "criteria": ["low", "high"]})
    assert score.keys == ["0", "1"]
    assert score.levels == ["low", "high"]


def test_locate_adapter_dir_is_a_decision_sidecar():
    """The locate adapter is a directory; a quantizer has to copy it as a tree."""
    from omlx.model_discovery import DECISION_SIDECAR_DIRS, PPLX_LOCATE_DIR

    assert PPLX_LOCATE_DIR in DECISION_SIDECAR_DIRS


# --------------------------------------------------------------------------
# locate (verbatim span extraction)


class StubTextModel(nn.Module):
    """Stand-in for ``backbone._text_model``: one Linear to wrap plus an embedding."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, HIDDEN)
        self.linear = nn.Linear(HIDDEN, HIDDEN)
        self.norm = nn.RMSNorm(HIDDEN)


def _locate_dir(
    model_dir,
    *,
    marker_id=900,
    alpha=4.0,
    rank=2,
    width=4,
    max_span=6,
    proposals=8,
    base=None,
    fmt=LOCATE_FORMAT,
    lora=None,
    targets=("linear",),
):
    directory = model_dir / LOCATE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "format": fmt,
        "marker_id": marker_id,
        "head": {"name": "span", "width": width, "max_span": max_span, "proposals": proposals},
        "lora": lora or {"alpha": alpha, "rank": rank, "layers": 0, "merged": False},
        "base": base or {},
        "thresholds": {"default": 0.5},
    }
    (directory / "locate_config.json").write_text(json.dumps(config))
    weights = {"language_model.embed_tokens.locate": mx.zeros((HIDDEN,))}
    for target in targets:
        weights[f"language_model.{target}.lora_a"] = mx.zeros((rank, HIDDEN))
        weights[f"language_model.{target}.lora_b"] = mx.zeros((HIDDEN, rank))
    mx.save_safetensors(str(directory / LOCATE_ADAPTER_WEIGHTS), weights)
    head = {}
    for name in ("ts", "te", "qs", "qe", "qp"):
        head[f"{name}.weight"] = mx.zeros((width, HIDDEN))
        head[f"{name}.bias"] = mx.zeros((width,))
    head["thresholds.weight"] = mx.zeros((3, HIDDEN))
    head["thresholds.bias"] = mx.zeros((3,))
    head["width_bias.weight"] = mx.zeros((32, 1))
    head["start_bias"] = mx.zeros((1,))
    head["end_bias"] = mx.zeros((1,))
    mx.save_safetensors(str(directory / LOCATE_HEAD_WEIGHTS), head)
    return directory


def test_normalize_question_accepts_locate():
    from omlx.models.pplx_decider import normalize_question

    question = normalize_question("q", {"type": "locate"})
    assert question.kind == "locate"
    assert question.keys == []
    assert question.texts == []
    with pytest.raises(DecisionRequestError, match="no criteria"):
        normalize_question("q", {"type": "locate", "criteria": ["a"]})


def test_locate_is_a_valid_question_type(tmp_path):
    model = _loaded_model(tmp_path)
    plan = model.encode({"state": "hello", "questions": {"q": {"type": "locate"}}})
    assert [item.kind for item in plan.locate] == ["locate"]
    # No decision chunk left to read out, so the decision pass is skipped.
    assert plan.sequences == []
    assert plan.state == "hello"


def test_locate_mixed_with_decisions_keeps_the_decisions(tmp_path):
    model = _loaded_model(tmp_path)
    plan = model.encode(
        {
            "state": "hello",
            "questions": {"q": {"type": "locate"}, "u": {"type": "noul"}},
        }
    )
    assert [item.kind for item in plan.locate] == ["locate"]
    assert plan.sequences[0].keys == ("u",)


def test_locate_rejects_more_than_the_question_limit(tmp_path):
    model = _loaded_model(tmp_path)
    questions = {f"q{index}": {"type": "locate"} for index in range(129)}
    with pytest.raises(DecisionRequestError, match="at most"):
        model.encode({"state": "hello", "questions": questions})


def test_locate_rejects_images(tmp_path):
    """The adapter was trained on a text state, so media is a request error."""
    model = _loaded_model(tmp_path)
    request = {"state": "hello", "questions": {"q": {"type": "locate"}}, "images": [_png_data_uri()]}
    with pytest.raises(DecisionRequestError, match="without images"):
        model.encode(request)
    mixed = {"state": "hello", "questions": {"q": {"type": "locate"}, "u": {"type": "noul"}},
             "images": [_png_data_uri()]}
    with pytest.raises(DecisionRequestError, match="without images"):
        model.encode(mixed)


def test_locate_adapter_loads_and_wraps_the_backbone(tmp_path):
    model = _loaded_model(tmp_path)
    _locate_dir(tmp_path / "checkpoint")
    adapter = model.locate
    assert model.locate_ready
    assert adapter.marker_id == 900
    assert [type(wrapper).__name__ for wrapper in adapter.adapters] == ["LocateLoRA"]
    assert adapter.head.max_span == 6
    # The marker joins the escapes, so user text cannot fake it.
    assert LOCATE_MARKER in model._escapes


def test_locate_adapter_rejects_a_foreign_marker_id(tmp_path):
    model = _loaded_model(tmp_path)
    _locate_dir(tmp_path / "checkpoint", marker_id=901)
    with pytest.raises(ValueError, match="trained with 901"):
        model.locate


def test_locate_adapter_rejects_a_foreign_base(tmp_path):
    model = _loaded_model(tmp_path)
    _locate_dir(
        tmp_path / "checkpoint",
        base={"sha256": {"readout.safetensors": "0" * 64}},
    )
    with pytest.raises(ValueError, match="another base"):
        model.locate


def test_locate_adapter_skips_the_quantized_config_in_the_base_check(tmp_path):
    """A quantizer rewrites config.json, so its hash cannot be verified."""
    model = _loaded_model(tmp_path)
    _locate_dir(
        tmp_path / "checkpoint",
        base={"sha256": {"config.json": "0" * 64, "readout.safetensors": _locate_sha256(tmp_path / "checkpoint" / "readout.safetensors")}},
    )
    assert model.locate.marker_id == 900


def test_locate_adapter_rejects_a_merged_or_partial_lora(tmp_path):
    model = _loaded_model(tmp_path)
    _locate_dir(tmp_path / "checkpoint", lora={"alpha": 4.0, "rank": 2, "layers": 3})
    with pytest.raises(ValueError, match="all-layer unmerged"):
        model.locate


def test_locate_adapter_rejects_an_unknown_format(tmp_path):
    model = _loaded_model(tmp_path)
    _locate_dir(tmp_path / "checkpoint", fmt="pplx-decider-locate-span-v0")
    with pytest.raises(ValueError, match="v0"):
        model.locate


def test_locate_adapter_rejects_unmatched_lora_tensors(tmp_path):
    model = _loaded_model(tmp_path)
    directory = _locate_dir(tmp_path / "checkpoint")
    weights = dict(mx.load(str(directory / LOCATE_ADAPTER_WEIGHTS)))
    weights["language_model.ghost.lora_a"] = mx.zeros((2, HIDDEN))
    weights["language_model.ghost.lora_b"] = mx.zeros((HIDDEN, 2))
    mx.save_safetensors(str(directory / LOCATE_ADAPTER_WEIGHTS), weights)
    with pytest.raises(ValueError, match="did not match"):
        model.locate


def test_locate_validate_rules():
    from omlx.models.pplx_decider import LocateAdapter

    validate = LocateAdapter.validate
    assert validate(None, "state", {"q": {"type": "locate"}}, 0) == [("q", LOCATE_CUTOFF)]
    assert validate(None, "state", {"q": {"type": "locate", "instructions": "x"}}, 0) == [
        ("x", LOCATE_CUTOFF)
    ]
    assert validate(None, "state", {"q": {"type": "locate", "instructions": "  "}}, 0) == [
        ("q", LOCATE_CUTOFF)
    ]
    assert validate(None, "state", {"q": {"type": "locate", "cutoff": 0.65}}, 0) == [("q", 0.65)]
    with pytest.raises(DecisionRequestError, match="without images"):
        validate(None, "state", {"q": {"type": "locate"}}, 1)
    with pytest.raises(DecisionRequestError, match="nonempty text state"):
        validate(None, "", {"q": {"type": "locate"}}, 0)
    with pytest.raises(DecisionRequestError, match="must be text or null"):
        validate(None, "state", {"q": {"type": "locate", "instructions": 7}}, 0)
    with pytest.raises(DecisionRequestError, match="at most"):
        validate(None, "state", {f"q{index}": {} for index in range(129)}, 0)


def test_locate_span_head_filters_by_width_and_threshold():
    from omlx.models.pplx_decider import LocateSpanHead

    head = LocateSpanHead(HIDDEN, 4, max_span=2)
    head.thresholds.bias = mx.zeros((3,))
    head.width_bias.weight = mx.zeros((32, 1))
    head.ts.weight, head.ts.bias = mx.zeros((4, HIDDEN)), mx.zeros((4,))
    head.te.weight, head.te.bias = mx.zeros((4, HIDDEN)), mx.zeros((4,))
    head.qs.weight, head.qs.bias = mx.zeros((4, HIDDEN)), mx.zeros((4,))
    head.qe.weight, head.qe.bias = mx.zeros((4, HIDDEN)), mx.zeros((4,))
    head.qp.weight, head.qp.bias = mx.zeros((4, HIDDEN)), mx.zeros((4,))
    document = mx.zeros((3, HIDDEN))
    labels = mx.zeros((2, HIDDEN))
    # Every pair scores exactly 0.5, so only the width and the cutoff filter.
    found = head.predict(document, labels, 0.5, proposals=8)
    assert [sorted(field) for field in found] == [
        [(0, 0, 0.5), (0, 1, 0.5), (1, 1, 0.5), (1, 2, 0.5), (2, 2, 0.5)],
        [(0, 0, 0.5), (0, 1, 0.5), (1, 1, 0.5), (1, 2, 0.5), (2, 2, 0.5)],
    ]
    assert head.predict(document, labels, 0.5001, proposals=8) == [[], []]


def test_locate_topk_is_descending_with_matching_indices():
    from omlx.models.pplx_decider import LocateSpanHead

    values, indices = LocateSpanHead._topk(mx.array([0.5, 2.0, 1.0, 3.0]), 2)
    assert values.tolist() == [3.0, 2.0]
    assert indices.tolist() == [3, 1]


def test_locate_lora_is_inert_until_enabled():
    from omlx.models.pplx_decider import LocateLoRA

    base = nn.Linear(HIDDEN, HIDDEN)
    wrapper = LocateLoRA(
        base, mx.full((2, HIDDEN), 0.5), mx.full((HIDDEN, 2), 2.0), alpha=4.0
    )
    inputs = mx.full((1, HIDDEN), 0.25)
    plain = base(inputs)
    assert mx.allclose(wrapper(inputs), plain).item()
    wrapper.enabled = True
    # Per element: sum_rank(inputs * lora_a) * lora_b * alpha / rank = 8 * 2 * 2.
    assert mx.allclose(wrapper(inputs), plain + 8.0).item()
    wrapper.enabled = False
    assert mx.allclose(wrapper(inputs), plain).item()


def test_locate_isolation_mask_blocks_questions_from_each_other():
    """The release layout: the context attends everything, a question only itself."""
    from omlx.models.pplx_decider import isolation_mask

    assert isolation_mask(3, [2]) is None
    allowed = isolation_mask(2, [2, 2])[0, 0].tolist()
    assert allowed[0] == [True] * 6 and allowed[1] == [True] * 6
    assert allowed[2] == [True, True, True, True, False, False]
    assert allowed[3] == [True, True, True, True, False, False]
    assert allowed[4] == [True, True, False, False, True, True]
    assert allowed[5] == [True, True, False, False, True, True]


def test_locate_positions_restart_in_every_question_block():
    from omlx.models.pplx_decider import LocateAdapter

    positions = LocateAdapter._positions(7, 3, [2, 2])[:, 0, :]
    assert positions.tolist() == [[0, 1, 2, 3, 4, 3, 4]] * 3


# --------------------------------------------------------------------------
# vision tower presence


def test_has_vision_module_sees_a_dropped_tower():
    """mlx-vlm sets a weightless top-level module to None; this must be seen."""
    assert has_vision_module(None) is False
    assert has_vision_module(nn.Linear(2, 2)) is False
    for name in ("vision_tower", "visual", "vision_model", "vision_encoder"):
        holder = nn.Linear(2, 2)
        setattr(holder, name, nn.Linear(2, 2))
        assert has_vision_module(holder) is True
    dropped = nn.Linear(2, 2)
    dropped.vision_tower = None
    assert has_vision_module(dropped) is False


def test_backbone_has_vision_needs_processor_and_tower():
    """An image processor alone does not make a checkpoint multimodal.

    mlx-vlm drops a module the checkpoint has no weights for, and the Qwen3.5
    path reads ``self.vision_tower.patch_embed`` without a None guard, so
    ``has_vision`` has to require the tower object too.
    """
    backbone = DecisionBackbone("nowhere")
    assert backbone.has_vision is False
    backbone.image_processor = object()
    assert backbone.has_vision is False
    backbone.model = nn.Linear(2, 2)
    assert backbone.has_vision is False
    backbone.model.vision_tower = nn.Linear(2, 2)
    assert backbone.has_vision is True
    backbone.image_processor = None
    assert backbone.has_vision is False


def test_locate_question_cutoff_is_validated(tmp_path):
    for bad in (0, 1.5, "0.5", True):
        with pytest.raises(DecisionRequestError, match="cutoff must be a number in"):
            normalize_question("q", {"type": "locate", "cutoff": bad})
    question = normalize_question("q", {"type": "locate", "cutoff": 0.65})
    assert question.cutoff == 0.65
    assert normalize_question("q", {"type": "locate"}).cutoff is None
    with pytest.raises(DecisionRequestError, match="cutoff is a locate option"):
        normalize_question("q", {"type": "noul", "cutoff": 0.5})


def test_locate_plan_carries_per_question_cutoffs(tmp_path):
    model = _loaded_model(tmp_path)
    plan = model.encode(
        {
            "state": "hello",
            "questions": {
                "a": {"type": "locate", "instructions": "x", "cutoff": 0.65},
                "b": {"type": "locate", "instructions": "y"},
            },
        }
    )
    assert [item.cutoff for item in plan.locate] == [0.65, None]


def test_locate_head_predict_takes_a_per_label_threshold():
    from omlx.models.pplx_decider import LocateSpanHead

    head = LocateSpanHead(4, 8, 16)
    document = mx.zeros((6, 4))
    labels = mx.zeros((2, 4))
    both = head.predict(document, labels, 0.5, proposals=8)
    assert len(both) == 2
    assert head.predict(document, labels, mx.array([0.5, 1.0]), proposals=8) == [
        both[0],
        [],
    ]
