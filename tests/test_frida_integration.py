# SPDX-License-Identifier: Apache-2.0
"""Real-weight parity against FRIDA-Decisions 0.4.0; opt-in local weights.

Download revision 3a7d751fc7d96c144ff1355d69516b68d751a6f7 and set
OMLX_FRIDA_MODEL_PATH to that directory. No download occurs during tests.
"""

import gc
import json
import os
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest
from frida_decisions.mlx_backend import MlxJudge
from frida_decisions.protocol import aggregate, decision

from omlx.models.frida import FridaModel

pytestmark = [pytest.mark.slow, pytest.mark.integration]
WEIGHTS_REVISION = "3a7d751fc7d96c144ff1355d69516b68d751a6f7"


@pytest.fixture
def model_path():
    value = os.environ.get("OMLX_FRIDA_MODEL_PATH")
    if not value or not (Path(value) / "model.safetensors").is_file():
        pytest.skip("Set OMLX_FRIDA_MODEL_PATH to the pinned original local checkpoint")
    path = Path(value)
    # snapshot_download(local_dir=...) records the exact downloaded revision.
    metadata = path / ".cache/huggingface/download/model.safetensors.metadata"
    if metadata.is_file():
        assert metadata.read_text().splitlines()[0] == WEIGHTS_REVISION
    return path


def cases():
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "Что хочет клиент?",
            "criteria": {
                "transfer": "Перенести номер от другого оператора",
                "refund": "Вернуть деньги",
                "cancel": "Отключить подписку",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Оцени срочность",
            "criteria": [
                "Не срочно",
                "Нужно ответить сегодня",
                "Авария, ответить немедленно",
            ],
        },
        "keep": {
            "type": "noul",
            "instructions": "Клиент хочет сохранить номер телефона?",
        },
        "rank": {
            "type": "ranking",
            "instructions": "Какие ответы помогут клиенту?",
            "criteria": {
                "port": "Подайте заявку на перенос номера к нам, номер сохранится.",
                "refund": "Для возврата денег нужен чек.",
                "subscribe": "Подключите платную подписку.",
            },
        },
    }
    state = "Хочу перейти к вам от другого оператора и сохранить мой номер. Что нужно сделать?"
    catalog = {
        f"id{i}": text
        for i, text in enumerate(
            [
                "Перенос номера",
                "Возврат оплаты",
                "Отключение подписки",
                "Ремонт телефона",
                "Замена сим карты",
                "Изменение тарифа",
                "Проверка баланса",
                "Пополнение счёта",
                "Оплата интернета",
                "Вызов мастера",
                "Жалоба на связь",
                "Смена адреса",
                "Доставка заказа",
                "Отмена заказа",
                "Проверка документов",
                "Получение выписки",
                "Открытие счёта",
                "Закрытие карты",
                "Восстановление пароля",
                "Регистрация номера",
            ]
        )
    }
    return [
        {"state": state, "questions": questions},
        {
            "state": {"message": state, "history": ["Здравствуйте", "Добрый день"]},
            "questions": questions,
        },
        {
            "state": state,
            "questions": {
                "catalog": {
                    "type": "ranking",
                    "instructions": "Какой материал подходит запросу?",
                    "criteria": catalog,
                }
            },
        },
        {
            "state": "длинный текст " * 400,
            "questions": {
                "long": {
                    "type": "ranking",
                    "instructions": "текст " * 120,
                    "criteria": ["вариант " * 300, {"answer": "короткий ответ"}],
                }
            },
        },
    ]


def drain(steps):
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_real_weight_adapter_upstream_and_packed_cached_parity(
    model_path, precision, record_property
):
    dtype = mx.float32 if precision == "fp32" else mx.bfloat16
    reference = MlxJudge.from_pretrained(
        model_path, dtype=dtype, state_max=384, rows_per_forward=1
    )
    expected = []
    worst_drift = 0.0
    for body in cases():
        parsed, candidates, tok = reference.compile(body)
        packed = reference._score_packed([tok])[0][0]
        reference.state_cache.clear()
        cached, _ = reference._score_cached(tok)
        # BF16 packed/cache paths in the pinned upstream have different
        # rounding. Require their decisions/order, and report margin drift.
        # Adapter-to-upstream comparisons below retain the requested tolerance.
        if precision == "fp32":
            np.testing.assert_allclose(packed, cached, atol=1e-3, rtol=1e-3)
        a, b = aggregate(parsed, candidates, packed), aggregate(
            parsed, candidates, cached
        )
        assert {k: decision(v) for k, v in a.items()} == {
            k: decision(v) for k, v in b.items()
        }
        for key in a:
            if a[key]["type"] == "ranking":
                assert a[key]["ranking"] == b[key]["ranking"]
        reference.state_cache.clear()
        margins, usage = reference._score([tok])
        expected.append(
            (
                margins[0],
                aggregate(parsed, candidates, margins[0]),
                usage["encoder_tokens"],
            )
        )
        worst_drift = max(
            worst_drift, float(np.max(np.abs(np.asarray(packed) - cached)))
        )
    reference.state_cache.clear()
    del reference
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    adapter = FridaModel(str(model_path), precision=precision)
    adapter.load()
    resident = mx.get_active_memory() - before
    loading_peak = mx.get_peak_memory() - before
    max_adapter_drift = 0.0
    decisions = []
    try:
        for body, (margins, answers, tokens) in zip(cases(), expected):
            plan = adapter.encode(body)
            observed = []

            def capture(parsed, candidates, values, observed=observed):
                observed.extend(values)
                return aggregate(parsed, candidates, values)

            with patch("frida_decisions.protocol.aggregate", capture):
                result = drain(adapter.run(plan, lambda: 1))
            np.testing.assert_allclose(observed, margins, atol=1e-3, rtol=1e-3)
            max_adapter_drift = max(
                max_adapter_drift, float(np.max(np.abs(np.asarray(observed) - margins)))
            )
            decisions.append({k: decision(v) for k, v in result["answers"].items()})
            assert {k: decision(v) for k, v in result["answers"].items()} == {
                k: decision(v) for k, v in answers.items()
            }
            for key in answers:
                if answers[key]["type"] == "ranking":
                    assert result["answers"][key]["ranking"] == answers[key]["ranking"]
            assert result["input_tokens"] == tokens
            assert result["usage"]["state_tokens"] == len(plan.tokenized.state)
            assert adapter.judge.state_cache is None
        record_property("max_adapter_upstream_margin_drift", max_adapter_drift)
        record_property("decisions", json.dumps(decisions, ensure_ascii=False))
        record_property("precision", precision)
        record_property("resident_bytes", resident)
        record_property("loading_peak_bytes", loading_peak)
        record_property("max_packed_cached_margin_drift", worst_drift)
        print(
            json.dumps(
                {
                    "precision": precision,
                    "resident_bytes": resident,
                    "loading_peak_bytes": loading_peak,
                    "max_packed_cached_margin_drift": worst_drift,
                }
            )
        )
    finally:
        adapter.close()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
    assert mx.get_active_memory() <= before + 2**20
