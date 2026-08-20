# SPDX-License-Identifier: Apache-2.0
"""The render/encode memo must be exact, bounded, and invisible on miss."""

from omlx.render_cache import RenderMemo, encode_cached


def test_same_inputs_same_key_different_inputs_different_key():
    base = ([{"role": "user", "content": "hi"}], None, {"a": 1}, False)
    assert RenderMemo.key(*base) == RenderMemo.key(*base)
    changed = ([{"role": "user", "content": "hi!"}], None, {"a": 1}, False)
    assert RenderMemo.key(*changed) != RenderMemo.key(*base)


def test_non_serializable_payload_disables_caching_not_rendering():
    class ImageHandle:
        pass

    assert RenderMemo.key([{"content": ImageHandle()}]) is None
    memo = RenderMemo()
    memo.put(None, "value")
    assert memo.get(None) is None
    assert memo.stats()["entries"] == 0


def test_lru_bounds_and_recency():
    memo = RenderMemo(max_entries=2)
    keys = [RenderMemo.key(str(i)) for i in range(3)]
    memo.put(keys[0], "a")
    memo.put(keys[1], "b")
    assert memo.get(keys[0]) == "a"  # refresh 0 so 1 is now oldest
    memo.put(keys[2], "c")
    assert memo.get(keys[1]) is None
    assert memo.get(keys[0]) == "a"
    assert memo.get(keys[2]) == "c"


def test_encode_cached_encodes_once_and_isolates_copies():
    class Tok:
        name_or_path = "test-tok"

        def __init__(self):
            self.calls = 0

        def encode(self, text):
            self.calls += 1
            return [7, 8, 9]

    tok = Tok()
    first = encode_cached(tok, "prompt")
    second = encode_cached(tok, "prompt")
    assert first == second == [7, 8, 9]
    assert tok.calls == 1
    first.append(0)
    assert encode_cached(tok, "prompt") == [7, 8, 9]


def test_encode_cached_shares_scope_across_deepcopy():
    """Scheduler.__init__ deep-copies the engine tokenizer ("Already
    borrowed" isolation); with the stamped scope the copy must share the
    memo — id()-based keys never match across a deepcopy."""
    import copy

    class Tok:
        name_or_path = "scoped-tok"
        _omlx_cache_id = "stable-scope"

        def __init__(self):
            self.calls = 0

        def encode(self, text):
            self.calls += 1
            return [4, 5, 6]

    original = Tok()
    clone = copy.deepcopy(original)
    assert encode_cached(original, "shared prompt") == [4, 5, 6]
    assert encode_cached(clone, "shared prompt") == [4, 5, 6]
    assert original.calls == 1
    assert clone.calls == 0, "the deep copy must hit the shared entry"


def test_encode_cached_distinguishes_tokenizers():
    class Tok:
        def __init__(self, name, ids):
            self.name_or_path = name
            self._ids = ids

        def encode(self, text):
            return list(self._ids)

    assert encode_cached(Tok("a", [1]), "same") == [1]
    assert encode_cached(Tok("b", [2]), "same") == [2]
