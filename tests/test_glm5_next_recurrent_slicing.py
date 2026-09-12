# SPDX-License-Identifier: Apache-2.0
"""A camada recorrente do glm5_next (Glm5NextLinearAttention) se fatia sozinha acima de 512 posições.

Medido em 04/09: um bloco de preparo de 2048 posições fica 30% mais rápido quando a recorrente
processa em fatias de 512 (bit a bit idêntico), e a atenção esparsa continua vendo o bloco
inteiro. A recursão tem que estar DENTRO do próprio `return` — é assim que a barreira de
ordenação do arquivo (test_glm5_next_ordering_barrier) a reconhece como saída legítima.
Lê o arquivo vendorado pelo caminho, como o teste irmão, para não depender do patch de compat.
"""
import ast
import pathlib

ALVO = pathlib.Path(__file__).resolve().parents[1] / (
    "omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py"
)


def _classe(nome):
    tree = ast.parse(ALVO.read_text(encoding="utf-8"))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == nome)


def test_o_limite_da_fatia_existe_e_e_512():
    classe = _classe("Glm5NextLinearAttention")
    limites = [
        n for n in classe.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_FATIA_RECORRENTE" for t in n.targets)
    ]
    assert limites and ast.literal_eval(limites[0].value) == 512


def test_a_chamada_fatia_por_recursao_dentro_do_return():
    classe = _classe("Glm5NextLinearAttention")
    call = next(n for n in classe.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    fatiamentos = [
        n for n in ast.walk(call)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        and "concatenate" in ast.unparse(n.value.func)
        and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "self" for c in ast.walk(n.value))
    ]
    assert fatiamentos, "o __call__ não fatia por recursão dentro de um return"
    guarda = next(
        n for n in ast.walk(call)
        if isinstance(n, ast.If) and "_FATIA_RECORRENTE" in ast.unparse(n.test)
    )
    assert "cache is not None" in ast.unparse(guarda.test), "o fatiamento tem que exigir cache"
