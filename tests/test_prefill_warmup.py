# SPDX-License-Identifier: Apache-2.0
"""O aquecimento do preparo depois da carga: ligado de fábrica, uma passagem, nada guardado.

Medido em 04/09 no GLM-5.3-Flash: o primeiro pedaço de 512 tokens depois de uma carga fria custa
~14,5 GB de memória do processo, os seguintes ~2 GB. Esse pedaço frio abortou o primeiro prompt
da produção em 2 de 4 cargas e fez o preditor recusar o bloco de 1024. O aquecimento paga esse
custo na carga, antes do tráfego.
"""

import types

import mlx.core as mx

from omlx.scheduler import Scheduler


def test_ligado_de_fabrica():
    from omlx.settings import ServerSettings

    assert ServerSettings().prefill_warmup is True
    assert ServerSettings.from_dict({}).prefill_warmup is True


def test_o_ajuste_desliga():
    from omlx.settings import ServerSettings

    assert ServerSettings.from_dict({"prefill_warmup": False}).prefill_warmup is False


class _Cache:
    def __init__(self):
        self.state = mx.zeros((1,))


class _Modelo:
    """Um modelo de mentira: registra a chamada e devolve caches simples."""

    def __init__(self):
        self.chamadas = []

    def make_cache(self):
        return [_Cache(), _Cache()]

    def __call__(self, ids, cache=None, **kwargs):
        self.chamadas.append((ids.shape, len(cache), dict(kwargs)))
        for c in cache:
            c.state = mx.ones((1,)) * ids.shape[1]
        return mx.zeros((1, ids.shape[1], 8))


def _agendador(modelo, skip=False):
    stub = types.SimpleNamespace(
        model=modelo,
        _stream=mx.default_stream(mx.cpu),
        _supports_skip_lm_head=lambda: skip,
    )
    return stub


def test_uma_passagem_do_tamanho_pedido_e_nada_fica_guardado():
    modelo = _Modelo()
    stub = _agendador(modelo)
    delta = Scheduler.warmup_prefill(stub, n_tokens=64)
    assert modelo.chamadas == [((1, 64), 2, {})]
    assert isinstance(delta, int) and delta >= 0
    # o cache é construído e descartado: nada pendurado no agendador
    assert not hasattr(stub, "cache") and not hasattr(stub, "_warmup_cache")


def test_pula_a_cabeca_de_vocabulario_quando_o_modelo_aceita():
    modelo = _Modelo()
    Scheduler.warmup_prefill(_agendador(modelo, skip=True), n_tokens=16)
    assert modelo.chamadas[0][2] == {"skip_lm_head": True}


def test_o_padrao_e_512_tokens():
    modelo = _Modelo()
    Scheduler.warmup_prefill(_agendador(modelo))
    assert modelo.chamadas[0][0] == (1, 512)
