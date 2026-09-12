# SPDX-License-Identifier: Apache-2.0
"""A torre de visão fora no uso só-texto: libera 1,12 GB no GLM-5.3-Flash e recusa imagem.

Medido em 05/09: um prompt de 229k tokens encostou no alvo do guarda perto dos 180k por ~0,4 GB e
pausou duas vezes. Desligado de fábrica — quem liga abre mão de mandar imagem para este modelo.
"""
import mlx.core as mx
import mlx.nn as nn

from omlx.utils.model_loading import free_vision_tower


class _Torre(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 64)
        self.norm = nn.LayerNorm(64)


class _Modelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _Torre()
        self.language_model = nn.Linear(64, 64)


def test_libera_so_a_torre_e_devolve_a_conta():
    m = _Modelo()
    mx.eval(m.parameters())
    antes_lm = m.language_model.weight
    n, nbytes = free_vision_tower(m)
    assert n == 4  # proj.weight, proj.bias, norm.weight, norm.bias
    assert nbytes == 64 * 64 * 4 + 64 * 4 + 64 * 4 + 64 * 4
    # a torre virou placeholders de um elemento, a árvore continua inteira
    assert m.vision_model.proj.weight.shape == (1,)
    assert m.vision_model.norm.bias.shape == (1,)
    # a linguagem não foi tocada
    assert m.language_model.weight is antes_lm


def test_desligado_de_fabrica():
    from omlx.settings import ServerSettings

    assert ServerSettings().vision_tower_text_only is False
    assert ServerSettings.from_dict({"vision_tower_text_only": True}).vision_tower_text_only is True
