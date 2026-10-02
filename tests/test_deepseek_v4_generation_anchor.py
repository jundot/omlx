# SPDX-License-Identifier: MIT
"""The generation anchor must survive system-final conversations.

encode_messages only emits ``<|Assistant|><think>`` after a user/developer
message, so two request shapes seen in production rendered with no anchor at
all and the model free-ran as document continuation until max_tokens:

  1. a system-only conversation (client title/summary utility prompts), and
  2. a trailing system message after the last user turn (workspace notes).
"""

from omlx.patches.deepseek_v4.chat_template_v4 import (
    ASSISTANT_SP_TOKEN,
    DS_TASK_SP_TOKENS,
    apply_chat_template,
    thinking_end_token,
    thinking_start_token,
)

THINK_ANCHOR = ASSISTANT_SP_TOKEN + thinking_start_token
CHAT_ANCHOR = ASSISTANT_SP_TOKEN + thinking_end_token

SYSTEM_ONLY = [
    {
        "role": "system",
        "content": "Generate a short session title from this conversation start.",
    }
]

TRAILING_SYSTEM = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Run the recon please."},
    {"role": "assistant", "content": "Working on it."},
    {"role": "user", "content": "continue"},
    {"role": "system", "content": "[Workspace::v1: /tmp/x files changed]"},
]

USER_FINAL = [
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Hello."},
]

TITLE_TASK = [{"role": "user", "content": "Summarize this.", "task": "title"}]
ACTION_TASK = [{"role": "user", "content": "Choose an action.", "task": "action"}]


def test_system_only_conversation_gets_anchor():
    out = apply_chat_template(SYSTEM_ONLY, add_generation_prompt=True)
    assert out.endswith(THINK_ANCHOR)


def test_trailing_system_after_user_gets_anchor():
    out = apply_chat_template(TRAILING_SYSTEM, add_generation_prompt=True)
    assert out.endswith(THINK_ANCHOR)


def test_chat_mode_appends_closed_anchor():
    out = apply_chat_template(
        SYSTEM_ONLY, add_generation_prompt=True, thinking_mode="chat"
    )
    assert out.endswith(CHAT_ANCHOR)
    assert not out.endswith(THINK_ANCHOR)


def test_user_final_rendering_unchanged():
    out = apply_chat_template(USER_FINAL, add_generation_prompt=True)
    assert out.endswith(THINK_ANCHOR)
    # Exactly one anchor: the guard must not double-append.
    assert out.count(THINK_ANCHOR) == 1


def test_title_task_transition_not_followed_by_generation_anchor():
    out = apply_chat_template(TITLE_TASK, add_generation_prompt=True)
    assert out.endswith(DS_TASK_SP_TOKENS["title"])
    assert not out.endswith(THINK_ANCHOR)


def test_action_task_transition_not_followed_by_second_generation_anchor():
    out = apply_chat_template(ACTION_TASK, add_generation_prompt=True)
    assert out.endswith(DS_TASK_SP_TOKENS["action"])
    assert out.count(THINK_ANCHOR) == 1


def test_no_anchor_without_generation_prompt():
    out = apply_chat_template(SYSTEM_ONLY, add_generation_prompt=False)
    assert not out.endswith(THINK_ANCHOR)
    assert not out.endswith(CHAT_ANCHOR)


def test_continue_final_message_not_anchored():
    msgs = USER_FINAL + [{"role": "assistant", "content": "Partial answer"}]
    out = apply_chat_template(msgs, continue_final_message=True)
    assert not out.endswith(THINK_ANCHOR)
    assert out.endswith("Partial answer")
