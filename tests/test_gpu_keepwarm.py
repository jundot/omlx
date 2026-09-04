# SPDX-License-Identifier: Apache-2.0
"""O keepwarm da GPU: desligado de fábrica, e o tique não muda estado nenhum."""

import os

import mlx.core as mx

from omlx.engine_core import EngineConfig, _keepwarm_tick


def test_ligado_de_fabrica():
    """Medido no servidor: +29% a +40% com pausa entre pedidos, custo zero em rajada."""
    assert EngineConfig().gpu_keepwarm is True
    from omlx.settings import ServerSettings

    assert ServerSettings().gpu_keepwarm is True


def test_o_ambiente_desliga():
    """OMLX_GPU_KEEPWARM=0 vence a config, para um A/B sem tocar no ajuste salvo."""
    from omlx.settings import ServerSettings

    assert ServerSettings.from_dict({"gpu_keepwarm": False}).gpu_keepwarm is False
    assert ServerSettings.from_dict({}).gpu_keepwarm is True


def test_o_tique_roda_e_nao_deixa_lixo():
    """Duas chamadas seguidas reaproveitam o mesmo escalar e não crescem a memória."""
    _keepwarm_tick()
    mx.synchronize()
    antes = mx.get_active_memory()
    for _ in range(20):
        _keepwarm_tick()
    mx.synchronize()
    depois = mx.get_active_memory()
    # 20 tiques de uma matriz 256x256 fp16 (128 KB) não podem mover o residente
    assert depois - antes < 8 * 1024 * 1024, f"{(depois - antes) / 1e6:.1f} MB a mais"


def test_o_tique_e_barato():
    """Ele existe para ocupar a GPU, não para trabalhar: tem que custar quase nada."""
    import time

    _keepwarm_tick()
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        _keepwarm_tick()
    mx.synchronize()
    ms = (time.perf_counter() - t0) * 1e3 / 50
    assert ms < 2.0, f"{ms:.2f} ms por tique — caro demais para um laço ocioso"
