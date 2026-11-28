# SPDX-License-Identifier: Apache-2.0
"""Offline SenseNova registration and small synthetic loading checks."""
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from omlx.api.openai_models import ChatCompletionRequest
from omlx.engine.base import GenerationOutput
from omlx.engine.batched import BatchedEngine
from omlx.model_discovery import detect_model_type
from omlx import server
from omlx.patches import sensenova_loop as registration
from omlx.utils.model_loading import maybe_apply_pre_load_patches


class TinyTokenizer:
    eos_token_ids = [1]
    eos_token_id = 1
    chat_template = None

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens, **kwargs):
        return "".join(chr(97 + token % 26) for token in tokens)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        # Restore process-global registration after each test, including failure cases.
        self.applied = registration._APPLIED
        self.module = sys.modules.get("mlx_lm.models.sensenova_loop")
        import mlx_lm.models
        self.models = mlx_lm.models
        self.attribute = getattr(self.models, "sensenova_loop", None)
        registration._APPLIED = False

    def tearDown(self):
        registration._APPLIED = self.applied
        if self.module is None:
            sys.modules.pop("mlx_lm.models.sensenova_loop", None)
        else:
            sys.modules["mlx_lm.models.sensenova_loop"] = self.module
        if self.attribute is None:
            if hasattr(self.models, "sensenova_loop"):
                del self.models.sensenova_loop
        else:
            self.models.sensenova_loop = self.attribute

    def require_support(self, fixture=False):
        try:
            available = importlib.util.find_spec("sensenova_mlx.registered_model") is not None
        except ModuleNotFoundError as exc:
            if exc.name != "sensenova_mlx":
                raise
            available = False
        if not available:
            self.skipTest("optional sensenova_mlx.registered_model is unavailable")
        if fixture:
            try:
                available = importlib.util.find_spec("tests.test_registered_model") is not None
            except ModuleNotFoundError as exc:
                if exc.name != "tests":
                    raise
                available = False
            if not available:
                self.skipTest("SenseNova project tests.test_registered_model fixture is unavailable")

    def test_registration_is_idempotent(self):
        self.require_support()
        from sensenova_mlx import registered_model
        self.assertFalse(registration.is_applied())
        self.assertTrue(registration.apply_sensenova_loop_patch())
        self.assertTrue(registration.is_applied())
        self.assertIs(sys.modules["mlx_lm.models.sensenova_loop"], registered_model)
        self.assertIs(self.models.sensenova_loop, registered_model)
        self.assertFalse(registration.apply_sensenova_loop_patch())

    def test_missing_package_and_unrelated_import_error(self):
        original = importlib.import_module
        for name in ("sensenova_mlx", "sensenova_mlx.registered_model", "other_dependency"):
            error = ModuleNotFoundError(f"missing {name}", name=name)
            def missing(module, *args, **kwargs):
                if module == "sensenova_mlx.registered_model":
                    raise error
                return original(module, *args, **kwargs)
            with patch.object(importlib, "import_module", side_effect=missing):
                if name == "other_dependency":
                    with self.assertRaises(ModuleNotFoundError) as caught:
                        registration.apply_sensenova_loop_patch()
                    self.assertIs(caught.exception, error)
                else:
                    with self.assertRaisesRegex(RuntimeError, "^SenseNova requires the separately supplied sensenova_mlx support package; see docs/sensenova.md$"):
                        registration.apply_sensenova_loop_patch()
            self.assertFalse(registration.is_applied())

    def test_unrelated_type_does_not_import_support(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
            with patch.object(registration, "apply_sensenova_loop_patch", side_effect=AssertionError("unexpected support import")):
                maybe_apply_pre_load_patches(str(root))
            self.assertFalse(registration.is_applied())

    def test_dispatch(self):
        self.require_support()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"model_type": "sensenova_loop"}))
            maybe_apply_pre_load_patches(str(root))
            from sensenova_mlx import registered_model
            self.assertIs(self.models.sensenova_loop, registered_model)
            self.assertIs(sys.modules["mlx_lm.models.sensenova_loop"], registered_model)
            self.assertTrue(registration.is_applied())

    def test_prefix_cache_guard_and_serial_admission(self):
        self.require_support(fixture=True)
        from sensenova_mlx.registered_model import Model, ModelArgs
        from tests.test_registered_model import tiny_raw
        from omlx.scheduler import Scheduler, SchedulerConfig
        with tempfile.TemporaryDirectory() as cache_dir:
            model = Model(ModelArgs.from_dict(tiny_raw()))
            scheduler = Scheduler(model, TinyTokenizer(), SchedulerConfig(
                max_num_seqs=8, paged_ssd_cache_dir=cache_dir, hot_cache_only=True,
                hot_cache_max_size=1024 * 1024, max_cache_blocks=8))
            try:
                self.assertTrue(scheduler._serialize_requests)
                self.assertEqual(scheduler._effective_max_num_seqs(), 1)
                self.assertIsNotNone(scheduler.block_aware_cache)
                self.assertTrue(scheduler._model_has_unreconstructible_cache())
            finally:
                scheduler.shutdown()

    def test_standard_load_and_short_synthetic_generation(self):
        self.require_support(fixture=True)
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from tests.test_registered_model import checkpoint
        from sensenova_mlx.registered_model import Model
        from omlx.scheduler import _serial_model_type

        previous = mx.default_device()
        mx.set_default_device(mx.cpu)
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, weights = checkpoint(root)
                tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(19)}, unk_token="0"))
                tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
                PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="0", eos_token="1").save_pretrained(root)
                maybe_apply_pre_load_patches(str(root))
                model, loaded_tokenizer = load(str(root), tokenizer_config={"local_files_only": True})
                self.assertIsInstance(model, Model)
                self.assertEqual(_serial_model_type(model), "sensenova_loop")
                self.assertEqual(set(dict(tree_flatten(model.parameters()))),
                                 set(weights) - {"model.visual.fixture", "model.language_model.loop_gate.fixture"})
                self.assertIsInstance(model.language_model.lm_head, nn.QuantizedLinear)
                self.assertEqual(loaded_tokenizer.encode("3 4"), [3, 4])
                tokens = list(generate_step(mx.array([3, 4]), model, stream=mx.default_stream(mx.cpu), max_tokens=2))
                self.assertEqual(len(tokens), 2)
                self.assertTrue(all(0 <= int(token.item() if hasattr(token, "item") else token) < 19
                                    for token, _ in tokens))
        finally:
            mx.set_default_device(previous)


class DiscoveryTests(unittest.TestCase):
    def test_real_config_shape_uses_text_only_route(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"model_type": "sensenova_loop", "architectures": ["SenseNovaLoopForConditionalGeneration"],
                      "text_config": {"model_type": "qwen3_5_text"}, "vision_config": {"model_type": "qwen3_5_vision"}}
            (root / "config.json").write_text(json.dumps(config))
            with self.assertLogs("omlx.model_discovery", level="WARNING"):
                self.assertEqual(detect_model_type(root), "llm")
            config["model_type"] = "SenseNova-Loop"
            (root / "config.json").write_text(json.dumps(config))
            with self.assertLogs("omlx.model_discovery", level="WARNING"):
                self.assertEqual(detect_model_type(root), "llm")


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_tool_arguments_rejected(self):
        with self.assertRaises(ValidationError):
            ChatCompletionRequest(model="sense", messages=[{"role": "assistant", "tool_calls": [{"function": {"name": "read_file", "arguments": "not json"}}]}])

    async def test_engine_chat_and_stream_pass_tools(self):
        from unittest.mock import AsyncMock
        engine = BatchedEngine("synthetic")
        engine._loaded = True
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        messages = [{"role": "tool", "tool_call_id": "call_1", "content": "result"}]
        output = GenerationOutput(text="done", new_text="done", finished=True, finish_reason="stop")
        async def fake_stream(**kwargs):
            self.assertEqual(kwargs["tools"], tools)
            yield output
        with patch.object(engine, "_preprocess_messages", side_effect=lambda m: m), \
             patch.object(engine, "_apply_chat_template", return_value="prompt") as template, \
             patch.object(engine, "_inject_specprefill_system_end"), \
             patch.object(engine, "_generation_prompt_text", return_value=(None, False)), \
             patch.object(engine, "_system_prefix_ids", return_value=None), \
             patch.object(engine, "generate", new=AsyncMock(return_value=output)) as generate, \
             patch.object(engine, "stream_generate", side_effect=fake_stream):
            self.assertIs(await engine.chat(messages, tools=tools), output)
            self.assertEqual(generate.await_args.kwargs["tools"], tools)
            self.assertEqual([item async for item in engine.stream_chat(messages, tools=tools)], [output])
            self.assertEqual(template.call_count, 2)
            self.assertTrue(template.call_args.args[1])

    async def test_generic_sse_with_fake_generation(self):
        class FakeEngine:
            tokenizer = TinyTokenizer()
            model_type = "sensenova_loop"
            async def stream_chat(self, messages, **kwargs):
                for text, done in [("<think>reason", False), ("</think>answer", True)]:
                    yield GenerationOutput(text=text, new_text=text, prompt_tokens=4,
                                           completion_tokens=2, finished=done,
                                           finish_reason="stop" if done else None)
        request = ChatCompletionRequest(model="sense", messages=[{"role":"user", "content":"hi"}],
                                        stream=True, stream_options={"include_usage":True})
        chunks = [chunk async for chunk in server.stream_chat_completion(FakeEngine(), [], request)]
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        events = [json.loads(c[6:]) for c in chunks[:-1]]
        choices = [c for event in events for c in event.get("choices", [])]
        self.assertEqual(choices[0]["delta"]["role"], "assistant")
        self.assertEqual("".join(c["delta"].get("reasoning_content", "") for c in choices), "reason")
        self.assertEqual("".join(c["delta"].get("content", "") for c in choices), "answer")
        self.assertEqual(choices[-1]["finish_reason"], "stop")
        usage = next(e["usage"] for e in events if e.get("usage"))
        self.assertEqual((usage["prompt_tokens"], usage["completion_tokens"]), (4, 2))


if __name__ == "__main__":
    unittest.main()
