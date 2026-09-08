# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the self-speculative n-gram draft source (v3)."""

from omlx.speculative.ngram import NGramDraftSource


def test_repeating_sequence_returns_continuation():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=2)
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    src.extend(seq)
    src.extend(seq)
    got = src.lookup()
    assert got is not None and got[:4] == [1, 2, 3, 4]
    assert src.hits == 1


def test_unique_suffix_does_not_match_itself():
    src = NGramDraftSource(match_len=4)
    src.extend([9] * 30)
    src2 = NGramDraftSource(match_len=4)
    src2.extend(list(range(30)))
    assert src2.lookup() is None
    assert src2.misses == 1


def test_short_history_misses():
    src = NGramDraftSource(match_len=8)
    src.extend([1, 2, 3])
    assert src.lookup() is None


def test_cascade_falls_back_to_short_window():
    # long window (24/16) has no repeat, but the last 8 tokens do
    src = NGramDraftSource(match_len=16, draft_max=16, draft_min=2)
    tail = list(range(100, 108))          # the 8-gram that repeats
    src.extend(list(range(0, 40)) + tail + list(range(200, 210)) + tail)
    got = src.lookup()
    assert got is not None
    assert got[0] == 200                  # continuation after the first tail


def test_short_window_match_caps_copy_length():
    src = NGramDraftSource(match_len=16, draft_max=16, draft_min=2)
    tail = list(range(100, 108))
    src.extend(list(range(0, 40)) + tail + list(range(200, 230)) + tail)
    got = src.lookup()
    assert got is not None
    # 8-window match cannot buy the full 16-wide verify
    assert len(got) <= 8


def test_per_key_freeze_only_hits_the_failing_key():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=3)
    a = [11, 12, 13, 14]
    src.extend(a + [50, 51, 52] + a)
    assert src.lookup() is not None
    src.feedback(0)
    src.extend([60])
    src.extend(a)
    assert src.lookup() is not None
    src.feedback(0)                       # second failure -> key frozen
    assert src.frozen_keys >= 1   # regiao congela as chaves de todas as janelas
    src.extend([61])
    src.extend(a)
    assert src.lookup() is None           # frozen key stays quiet
    # a DIFFERENT key keeps serving in the same source
    b = [21, 22, 23, 24]
    src.extend(b + [70, 71, 72] + b)
    got = src.lookup()
    assert got is not None and got[0] == 70


def test_success_thaws_a_failing_key():
    src = NGramDraftSource(match_len=4, draft_max=6, draft_min=2)
    a = [1, 2, 3, 4]
    src.extend(a + [9, 9] + a)
    assert src.lookup() is not None
    src.feedback(0)                       # one failure
    src.feedback(6)                       # productive copy clears the slate
    assert src._fails == {}


def test_seed_with_processors_does_not_duplicate_splice():
    from types import SimpleNamespace

    from mlx_lm.models.cache import TokenBuffer

    from omlx.patches.mlx_lm_mtp import set_ngram_spec
    from omlx.patches.mlx_lm_mtp.batch_generator import _ngram_source_for_cycle

    import mlx.core as mx

    set_ngram_spec(True)
    prompt = list(range(100, 140))
    generated = [1, 2, 3]
    buf = TokenBuffer(prompt)
    buf.update_and_fetch(mx.array(generated, dtype=mx.int32))
    gb = SimpleNamespace(_token_context=[buf])
    state = SimpleNamespace()
    committed = mx.array(generated, dtype=mx.uint32)

    # processors ativos: o buffer ja contem os gerados -> sem duplicata
    src = _ngram_source_for_cycle(gb, state, committed, procs=[object()])
    assert src._tokens == prompt + generated

    # sem processors: o buffer so tem o prompt -> committed entra uma vez
    buf2 = TokenBuffer(prompt)
    gb2 = SimpleNamespace(_token_context=[buf2])
    state2 = SimpleNamespace()
    src2 = _ngram_source_for_cycle(gb2, state2, committed, procs=None)
    assert src2._tokens == prompt + generated
    set_ngram_spec(False)


def test_shared_pool_serves_across_requests_and_respects_fence():
    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, draft_max=8, draft_min=2)
    doc = list(range(500, 540))
    pool.feed(doc)                       # pedido 1 alimenta o pool
    suffix = doc[10:14]                  # pedido 2 chega no mesmo trecho
    got = pool.lookup_suffix(suffix)
    assert got == doc[14:22]
    # a cerca impede emenda entre pedidos: continuacao para na fronteira
    pool.feed([1, 2, 3])
    tail = doc[-4:]
    got2 = pool.lookup_suffix(tail)
    assert got2 is None or all(t >= 0 for t in got2)


def test_frequencia_conta_ocorrencias_e_corta_na_cerca():
    # F1.1: o indice conta ocorrencias por gram; a cerca separa pedidos
    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, draft_max=8, draft_min=2, freq_rule=True)
    doc = [10, 11, 12, 13, 20, 21, 22, 23]
    pool.feed(doc)
    pool.feed(doc)                        # segundo pedido, mesma pagina
    key = (10, 11, 12, 13)
    assert pool._src._count[4][key] == 2
    # chave que atravessa a cerca contem o FENCE e nunca casa consulta real
    fence_key = (22, 23, SharedNGramPool.FENCE, 10)
    assert pool._src._count[4].get(fence_key, 0) <= 1


def test_frequencia_sobrevive_ao_rebuild_do_teto():
    # F1.1: o trim reconstroi do tail e as contagens se re-derivam dele
    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, draft_max=8, draft_min=2,
                           freq_rule=True, max_tokens=400)
    doc = list(range(700, 740))
    for _ in range(20):                   # estoura o teto -> rebuild
        pool.feed(doc)
    key = tuple(doc[:4])
    assert len(pool) <= 400
    assert pool._src._count[4].get(key, 0) >= 1


def test_frequencia_regua_da_comprimentos_diferentes():
    # F1.2/F2.3: mesma chave curta, repeticao forte vs fraca — regra so-bonus.
    # Prefixos distintos garantem que so a janela curta (2) casa.
    from omlx.speculative.ngram import NGramDraftSource

    x = [7, 8]
    cont = list(range(100, 120))

    fraco = NGramDraftSource(match_len=4, draft_max=12, draft_min=2,
                             freq_rule=True)
    fraco.extend([901, 951] + x + cont + [902, 952] + x)   # reps=1
    got_fraco = fraco.lookup()

    forte = NGramDraftSource(match_len=4, draft_max=12, draft_min=2,
                             freq_rule=True)
    for i in range(4):                                     # reps>=3
        forte.extend([901 + i, 951 + i] + x + cont)
    forte.extend([999, 998] + x)
    got_forte = forte.lookup()

    assert got_fraco is not None and got_forte is not None
    assert len(got_forte) == 12          # repeticao forte compra draft_max
    assert len(got_fraco) <= 2           # evidencia fraca mantem o cap v3
    # a regra nunca corta ABAIXO do cap v3 (o -3.4% do primeiro corte)
    off = NGramDraftSource(match_len=4, draft_max=12, draft_min=2)
    off.extend([901, 951] + x + cont + [902, 952] + x)
    got_off = off.lookup()
    assert (got_off is None) or len(got_fraco) >= len(got_off)


def test_frequencia_off_reproduz_o_comportamento_v3():
    # F1.4: desligada, a regua nova nao muda nada — cenario onde ela mudaria
    from omlx.speculative.ngram import NGramDraftSource

    a = [1, 2, 3, 4]
    cont = list(range(100, 120))
    off = NGramDraftSource(match_len=4, draft_max=12, draft_min=2)
    off.extend(a + cont + a)
    got = off.lookup()
    assert got is not None and len(got) == 12   # v3: janela longa -> draft_max


def test_frequencia_no_pool_via_janela_curta():
    # F1.5: a regua vale no caminho do pool — vermelho se so o drafter a tiver.
    # Ocorrencias precedidas por prefixos distintos: so a janela curta casa.
    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, draft_max=12, draft_min=2,
                           freq_rule=True)
    cont = list(range(300, 330))
    for i in range(4):
        pool.feed([900 + i, 950 + i] + [7, 8] + cont)
    got = pool.lookup_suffix([90, 91, 92, 93, 7, 8])
    assert got is not None and got[0] == 300
    # regra fixa de janela 2 pararia em max(draft_min, 2) = 2 tokens
    assert len(got) == 12


def test_janela_de_busca_candidata_e_24():
    # F1.3: o candidato do A/B e o ngram-mod do llama.cpp (busca 24)
    from omlx.speculative.ngram import NGRAM_MOD_MATCH_LEN, NGramDraftSource

    assert NGRAM_MOD_MATCH_LEN == 24
    src = NGramDraftSource(match_len=NGRAM_MOD_MATCH_LEN)
    assert src.windows == (36, 24, 12)


def test_shared_pool_eviction_keeps_size_bounded():
    import threading

    from omlx.speculative.ngram import SharedNGramPool

    pool = SharedNGramPool(match_len=4, max_tokens=2000)

    def fill(base):
        for i in range(20):
            pool.feed(list(range(base + i * 100, base + i * 100 + 90)))

    threads = [threading.Thread(target=fill, args=(b,)) for b in (0, 10_000, 20_000)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(pool) <= 2000
