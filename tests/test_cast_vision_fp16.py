# SPDX-License-Identifier: Apache-2.0
"""O cast da torre de visão na carga: só as folhas float32 da subárvore de visão viram float16.

Medido em 04/09 no GLM-5.3-Flash (visao_f16.py, só a torre): float16 fica 4-5x mais perto do
float32 do que bfloat16 (o dtype de origem), pico de ativação ~228 contra 65504. O float32 que a
rotina oQ grava por regra do tronco (#1682, medido no Gemma 4) custa ~1 GB residente à toa.
"""
import mlx.core as mx
import mlx.nn as nn

from omlx.utils.model_loading import cast_vision_f32_params_to_fp16


class _Torre(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.proj.weight = mx.ones((4, 4), dtype=mx.float32)
        self.proj.bias = mx.zeros((4,), dtype=mx.float32)
        self.norm = mx.ones((4,), dtype=mx.float16)   # já float16: fica


class _Lingua(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = mx.ones((3, 4), dtype=mx.float32)   # roteador em float32: NÃO é visão, fica


class _VLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = _Torre()
        self.language_model = _Lingua()


def test_so_as_folhas_float32_da_visao_viram_float16():
    m = _VLM()
    n = cast_vision_f32_params_to_fp16(m)
    assert n == 2, "proj.weight e proj.bias; a norma já era float16"
    assert m.vision_model.proj.weight.dtype == mx.float16
    assert m.vision_model.proj.bias.dtype == mx.float16
    assert m.vision_model.norm.dtype == mx.float16
    assert m.language_model.gate.dtype == mx.float32, "o roteador da língua não é tocado"
    assert float(mx.sum(m.vision_model.proj.weight).item()) == 16.0


def test_sem_visao_nao_faz_nada():
    class _SoTexto(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = _Lingua()
    assert cast_vision_f32_params_to_fp16(_SoTexto()) == 0
