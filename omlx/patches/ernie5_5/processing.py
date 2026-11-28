# SPDX-License-Identifier: Apache-2.0

TOOL_CALL_START = "<ernie:tool_call>"
TOOL_CALL_END = "</ernie:tool_call>"
_RENDER_OPTIONS = ("system_settings", "agent_storage", "examples", "options")


class TokenizedPrompt(str):
    """A rendered chat prompt with the token ids its renderer encoded."""

    token_ids: tuple[int, ...]

    def __new__(cls, text, token_ids):
        prompt = super().__new__(cls, text)
        prompt.token_ids = tuple(token_ids)
        return prompt

    def __getnewargs__(self):
        return str(self), self.token_ids


class Processor:
    """Render chats with the release's Python renderer and segment encoder.

    Only renderer markers become control tokens, so caller text that spells
    a marker stays ordinary text.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        tokenizer.apply_chat_template = self.apply_chat_template
        tokenizer.has_tool_calling = True
        tokenizer.tool_call_start = TOOL_CALL_START
        tokenizer.tool_call_end = TOOL_CALL_END
        tokenizer.tool_parser = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        continue_final_message=False,
        tools=None,
        enable_thinking=None,
        reasoning_effort=None,
        **kwargs,
    ):
        options = {key: kwargs[key] for key in _RENDER_OPTIONS if key in kwargs}
        if reasoning_effort is not None:
            options["reasoning_effort"] = reasoning_effort
        rendered = self.tokenizer.render_chat(
            messages,
            tools,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking,
            **options,
        )
        token_ids = self.tokenizer.encode_segments(rendered.segments)
        return token_ids if tokenize else TokenizedPrompt(rendered.text, token_ids)
