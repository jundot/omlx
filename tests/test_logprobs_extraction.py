# SPDX-License-Identifier: Apache-2.0
"""Tests for engine-side per-token logprob extraction (#1549, Phase 2a)."""

import mlx.core as mx

from omlx.logprobs import extract_token_logprob
from omlx.request import RequestOutput, TokenLogprob

# vocab=5 logprob vector; descending order by value is idx 1, 4, 2, 0, 3
LP = mx.array([-3.0, -0.1, -2.0, -5.0, -1.0])


class TestExtractTokenLogprob:
    def test_returns_token_logprob_type(self):
        assert isinstance(extract_token_logprob(LP, 1, 2), TokenLogprob)

    def test_chosen_logprob(self):
        tl = extract_token_logprob(LP, chosen_token_id=1, top_k=0)
        assert tl.token_id == 1
        assert abs(tl.logprob - (-0.1)) < 1e-5

    def test_chosen_can_differ_from_argmax(self):
        # chosen token 0 (logprob -3.0) even though argmax is 1
        tl = extract_token_logprob(LP, chosen_token_id=0, top_k=0)
        assert tl.token_id == 0
        assert abs(tl.logprob - (-3.0)) < 1e-5

    def test_topk_ids_sorted_descending(self):
        tl = extract_token_logprob(LP, chosen_token_id=1, top_k=3)
        assert tl.top_ids == [1, 4, 2]
        assert all(
            a >= b for a, b in zip(tl.top_logprobs, tl.top_logprobs[1:])
        )

    def test_topk_logprobs_match_ids(self):
        tl = extract_token_logprob(LP, chosen_token_id=1, top_k=2)
        assert tl.top_ids == [1, 4]
        assert abs(tl.top_logprobs[0] - (-0.1)) < 1e-5
        assert abs(tl.top_logprobs[1] - (-1.0)) < 1e-5

    def test_topk_zero_returns_empty_top_but_keeps_chosen(self):
        tl = extract_token_logprob(LP, chosen_token_id=2, top_k=0)
        assert tl.top_ids == []
        assert tl.top_logprobs == []
        assert abs(tl.logprob - (-2.0)) < 1e-5

    def test_topk_clamped_to_vocab(self):
        tl = extract_token_logprob(LP, chosen_token_id=1, top_k=100)
        assert len(tl.top_ids) == 5
        assert tl.top_ids[0] == 1  # still sorted desc


class TestRequestOutputLogprobsField:
    def test_default_none(self):
        out = RequestOutput(request_id="r1")
        assert out.logprobs is None

    def test_accepts_list(self):
        tl = TokenLogprob(token_id=1, logprob=-0.1, top_ids=[1], top_logprobs=[-0.1])
        out = RequestOutput(request_id="r1", logprobs=[tl])
        assert out.logprobs[0].token_id == 1


class TestOutputCollectorLogprobsMerge:
    """_merge_outputs must accumulate per-token logprobs across steps (#1549)."""

    def test_merge_concatenates_logprobs(self):
        from omlx.output_collector import RequestOutputCollector

        c = RequestOutputCollector(aggregate=True)
        t1 = TokenLogprob(token_id=1, logprob=-0.1)
        t2 = TokenLogprob(token_id=2, logprob=-0.2)
        c.put(RequestOutput(request_id="r", new_token_ids=[1], logprobs=[t1]))
        c.put(RequestOutput(request_id="r", new_token_ids=[2], logprobs=[t2]))
        out = c.get_nowait()
        assert [tl.token_id for tl in out.logprobs] == [1, 2]

    def test_merge_one_side_none(self):
        from omlx.output_collector import RequestOutputCollector

        c = RequestOutputCollector(aggregate=True)
        t1 = TokenLogprob(token_id=1, logprob=-0.1)
        c.put(RequestOutput(request_id="r", new_token_ids=[1], logprobs=[t1]))
        c.put(RequestOutput(request_id="r", new_token_ids=[2], logprobs=None))
        out = c.get_nowait()
        assert [tl.token_id for tl in out.logprobs] == [1]

    def test_merge_both_none_stays_none(self):
        from omlx.output_collector import RequestOutputCollector

        c = RequestOutputCollector(aggregate=True)
        c.put(RequestOutput(request_id="r", new_token_ids=[1]))
        c.put(RequestOutput(request_id="r", new_token_ids=[2]))
        out = c.get_nowait()
        assert out.logprobs is None


class TestGenerationOutputLogprobs:
    def test_default_none(self):
        from omlx.engine.base import GenerationOutput

        assert GenerationOutput(text="hi").logprobs is None

    def test_accepts_logprobs(self):
        from omlx.engine.base import GenerationOutput

        tl = TokenLogprob(token_id=1, logprob=-0.1)
        go = GenerationOutput(text="hi", logprobs=[tl])
        assert go.logprobs[0].token_id == 1


class _FakeTokenizer:
    def __init__(self, mapping):
        self._m = mapping

    def decode(self, ids):
        return "".join(self._m.get(int(i), "?") for i in ids)


class TestBuildChoiceLogprobs:
    """OpenAI-shape formatting from engine TokenLogprob (#1549, Phase 3)."""

    def test_none_and_empty_return_none(self):
        from omlx.api.utils import build_choice_logprobs

        assert build_choice_logprobs(None, None) is None
        assert build_choice_logprobs([], None) is None

    def test_token_and_bytes_decoded(self):
        from omlx.api.utils import build_choice_logprobs

        tok = _FakeTokenizer({1: "Hi", 4: " Hello", 2: "x"})
        tls = [
            TokenLogprob(
                token_id=1, logprob=-0.1, top_ids=[1, 4], top_logprobs=[-0.1, -1.0]
            )
        ]
        cl = build_choice_logprobs(tls, tok)
        entry = cl.content[0]
        assert entry.token == "Hi"
        assert entry.logprob == -0.1
        assert entry.bytes == list("Hi".encode("utf-8"))
        assert [t.token for t in entry.top_logprobs] == ["Hi", " Hello"]
        assert entry.top_logprobs[1].bytes == list(" Hello".encode("utf-8"))

    def test_top_logprobs_zero_yields_empty_top_list(self):
        from omlx.api.utils import build_choice_logprobs

        tok = _FakeTokenizer({2: "x"})
        tls = [TokenLogprob(token_id=2, logprob=-0.5, top_ids=[], top_logprobs=[])]
        cl = build_choice_logprobs(tls, tok)
        assert cl.content[0].token == "x"
        assert cl.content[0].top_logprobs == []

    def test_one_entry_per_token(self):
        from omlx.api.utils import build_choice_logprobs

        tok = _FakeTokenizer({1: "a", 2: "b", 3: "c"})
        tls = [
            TokenLogprob(token_id=1, logprob=-0.1),
            TokenLogprob(token_id=2, logprob=-0.2),
            TokenLogprob(token_id=3, logprob=-0.3),
        ]
        cl = build_choice_logprobs(tls, tok)
        assert [e.token for e in cl.content] == ["a", "b", "c"]


def _lp(tid, text):
    return TokenLogprob(token_id=tid, logprob=-0.1 * tid, text=text)


class TestThinkingParserLogprobAlignment:
    """Streaming logprob/content-token alignment under buffering (#1549, Phase 3b)."""

    def test_text_only_feed_unchanged(self):
        from omlx.api.thinking import ThinkingParser

        p = ThinkingParser()
        assert p.feed("<think>hi</think>yo") == ("hi", "yo")
        assert p.finish() == ("", "")

    def test_angle_bracket_in_content_keeps_all_logprobs(self):
        # The drift case: "a", "<", "b" — the '<' triggers tag-lookahead buffering.
        from omlx.api.thinking import ThinkingParser

        p = ThinkingParser()
        td, cd, clps = p.feed_with_logprob("a", _lp(1, "a"))
        assert cd == "a" and [x.token_id for x in clps] == [1]
        td, cd, clps = p.feed_with_logprob("<", _lp(2, "<"))
        assert cd == "" and clps == []  # buffered, nothing emitted yet
        td, cd, clps = p.feed_with_logprob("b", _lp(3, "b"))
        assert cd == "<b" and [x.token_id for x in clps] == [2, 3]
        # Total content entries (1 + 0 + 2) == 3 content tokens. Aligned.

    def test_special_token_thinking_tokens_excluded(self):
        from omlx.api.thinking import ThinkingParser

        p = ThinkingParser()
        td, cd, clps = p.feed_with_logprob("<think>", _lp(1, "<think>"))
        assert cd == "" and clps == []
        td, cd, clps = p.feed_with_logprob("reason", _lp(2, "reason"))
        assert td == "reason" and cd == "" and clps == []
        td, cd, clps = p.feed_with_logprob("</think>", _lp(3, "</think>"))
        assert cd == "" and clps == []
        td, cd, clps = p.feed_with_logprob("ans", _lp(4, "ans"))
        assert cd == "ans" and [x.token_id for x in clps] == [4]

    def test_finish_flushes_buffered_content_logprob(self):
        from omlx.api.thinking import ThinkingParser

        p = ThinkingParser()
        td, cd, clps = p.feed_with_logprob("<", _lp(1, "<"))
        assert cd == ""
        td, cd, clps = p.finish_with_logprob()
        assert cd == "<" and [x.token_id for x in clps] == [1]

    def test_split_tag_across_tokens_drops_tag_keeps_content(self):
        from omlx.api.thinking import ThinkingParser

        p = ThinkingParser()
        # "<", "think>" arriving as two tokens => a real open tag, no content
        td, cd, clps = p.feed_with_logprob("<", _lp(1, "<"))
        assert cd == "" and clps == []
        td, cd, clps = p.feed_with_logprob("think>", _lp(2, "think>"))
        assert cd == "" and clps == []  # completed <think>, consumed
        td, cd, clps = p.feed_with_logprob("hi", _lp(3, "hi"))
        assert td == "hi" and cd == "" and clps == []
