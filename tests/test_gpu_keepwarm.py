# SPDX-License-Identifier: Apache-2.0
"""O keepwarm da GPU: desligado de fábrica, e o tique não muda estado nenhum."""

import os

import mlx.core as mx

from omlx.engine_core import EngineConfig, _keepwarm_tick


def test_desligado_de_fabrica():
    """Ninguém paga por ele sem pedir — nem pela config, nem pelo ambiente."""
    assert EngineConfig().gpu_keepwarm is False
    assert os.environ.get("OMLX_GPU_KEEPWARM") != "1"


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
