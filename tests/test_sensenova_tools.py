# SPDX-License-Identifier: Apache-2.0
"""Optional offline tokenizer and synthetic tool-stream fixture; no weights are loaded."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import unittest

from mlx_lm.utils import load_tokenizer
from omlx import server
from omlx.api import thinking, tool_calling, utils as api_utils
from omlx.api.openai_models import ChatCompletionRequest, Message
from omlx.engine.base import GenerationOutput

TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
USER = "Call read_file for example.txt."
CALL = "<tool_call>\n<function=read_file>\n<parameter=path>\nexample.txt\n</parameter>\n</function>\n</tool_call>"


@unittest.skipUnless(os.environ.get("SENSENOVA_MODEL_PATH"), "set SENSENOVA_MODEL_PATH to a local tokenizer artifact")
@unittest.skipUnless(importlib.util.find_spec("sensenova_mlx"), "optional SenseNova support is not installed")
class ActualToolTemplate(unittest.TestCase):
    def test_synthetic_tool_cycle(self):
        artifact = Path(os.environ["SENSENOVA_MODEL_PATH"])
        eos = json.loads((artifact / "generation_config.json").read_text())["eos_token_id"]
        self.assertEqual(eos, [248044, 248046])
        tokenizer = load_tokenizer(str(artifact), tokenizer_config_extra={"trust_remote_code": False, "local_files_only": True}, eos_token_ids=eos)
        self.assertEqual(set(tokenizer.eos_token_ids), set(eos))
        initial_messages = [{"role": "user", "content": USER}]
        initial = tokenizer.apply_chat_template(initial_messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        self.assertIn("<tool_call>\n<function=example_function_name>", initial)
        self.assertTrue(initial.endswith("<|im_start|>assistant\n<think>\n"))

        response = "I should read the file.\n</think>\n" + CALL
        thought, regular = thinking.extract_thinking(response, start_in_thinking=True)
        extraction = tool_calling.extract_tool_calls_with_thinking(thought, regular, tokenizer, TOOLS, finish_reason="stop")
        self.assertEqual(thought, "I should read the file.")
        self.assertEqual(extraction.cleaned_thinking, thought)
        self.assertEqual(extraction.cleaned_text, "")
        self.assertEqual(extraction.parse_errors, ())
        self.assertEqual(len(extraction.tool_calls or []), 1)
        call = extraction.tool_calls[0]
        self.assertTrue(call.id)
        self.assertEqual(call.type, "function")
        self.assertEqual(call.function.name, "read_file")
        self.assertEqual(json.loads(call.function.arguments), {"path": "example.txt"})

        chunks = [CALL[:4], CALL[4:31], CALL[31:65], CALL[65:]]
        stream_filter = tool_calling.ToolCallStreamFilter(tokenizer, tools=TOOLS, capture_ordered_segments=True)
        visible = "".join(stream_filter.feed(chunk) for chunk in chunks) + stream_filter.finish()
        self.assertEqual(visible, "")
        self.assertEqual(stream_filter.take_completed_envelopes(), [CALL])

        messages = [Message(role="user", content=USER), Message(role="assistant", content=None, reasoning_content=thought, tool_calls=[call.model_dump()]), Message(role="tool", tool_call_id=call.id, content="example content")]
        prepared = api_utils.extract_text_content(messages, tokenizer=tokenizer, native_reasoning_content=True)
        self.assertEqual(prepared[-1]["tool_call_id"], call.id)
        self.assertEqual(prepared[1]["tool_calls"][0]["function"]["arguments"], {"path": "example.txt"})
        replay = tokenizer.apply_chat_template(prepared, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        self.assertIn(CALL, replay)
        self.assertIn("<|im_start|>user\n<tool_response>\nexample content\n</tool_response><|im_end|>", replay)
        self.assertTrue(replay.endswith("<|im_start|>assistant\n<think>\n"))
        self.assertNotIn(call.id, replay)  # The API associates the single result, not the template.
        request = ChatCompletionRequest(model="sense", messages=messages, tools=TOOLS, tool_choice="auto", stream=True)

        class FakeEngine:
            model_type = "sensenova_loop"
            async def stream_chat(self, messages, **kwargs):
                text = "<think>" + response
                for chunk in (text[:20], text[20:50], text[50:]):
                    yield GenerationOutput(text=chunk, new_text=chunk, prompt_tokens=4,
                                           completion_tokens=3, finished=chunk == text[50:],
                                           finish_reason="stop" if chunk == text[50:] else None)

        engine = FakeEngine()
        engine.tokenizer = tokenizer

        async def stream():
            return [chunk async for chunk in server.stream_chat_completion(engine, prepared, request, tools=TOOLS)]

        chunks = asyncio.run(stream())
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        events = [json.loads(chunk[6:]) for chunk in chunks[:-1]]
        choices = [choice for event in events for choice in event.get("choices", [])]
        calls = [tool for choice in choices for tool in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "example.txt"})
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")
        self.assertNotIn("<tool_call>", "".join(choice.get("delta", {}).get("content", "") or "" for choice in choices))


if __name__ == "__main__":
    unittest.main(verbosity=2)
