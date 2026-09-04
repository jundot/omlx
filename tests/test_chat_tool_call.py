# SPDX-License-Identifier: Apache-2.0
"""Tests for chat MCP tool call loop (chat.html streamResponse changes)."""

import json
from pathlib import Path

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx" / "admin" / "templates" / "chat.html"


class TestChatToolCallMessageFiltering:
    """Test the messagesForApi filtering logic (Python equivalent of the JS)."""

    @staticmethod
    def build_messages_for_api(messages):
        """Replicate the messagesForApi logic from streamResponse in chat.html."""
        valid_roles = {"user", "assistant", "tool", "system"}
        result = []
        for msg in messages:
            if msg["role"] not in valid_roles:
                continue
            m = {"role": msg["role"], "content": msg.get("content")}
            if msg.get("tool_calls"):
                m["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                m["tool_call_id"] = msg["tool_call_id"]
            result.append(m)
        return result

    def test_filters_tool_call_indicator_messages(self):
        """tool_call role messages must not be sent to the API."""
        messages = [
            {"role": "user", "content": "Who is X?"},
            {"role": "tool_call", "content": "tavily__tavily_search…", "_ui": True},
            {"role": "assistant", "content": "X is...", "tool_calls": None},
        ]
        api_msgs = self.build_messages_for_api(messages)
        roles = [m["role"] for m in api_msgs]
        assert "tool_call" not in roles
        assert roles == ["user", "assistant"]

    def test_passes_tool_calls_and_tool_call_id(self):
        """Assistant tool_calls and tool result tool_call_id must be preserved."""
        tc = [{"id": "tc_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        messages = [
            {"role": "user", "content": "Search for X"},
            {"role": "assistant", "content": None, "tool_calls": tc, "_ui": False},
            {"role": "tool", "tool_call_id": "tc_1", "content": "result...", "_ui": False},
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 3
        assert api_msgs[1]["tool_calls"] == tc
        assert api_msgs[2]["tool_call_id"] == "tc_1"

    def test_normal_conversation_unchanged(self):
        """Normal user/assistant conversation with no tools is unaffected."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        api_msgs = self.build_messages_for_api(messages)
        assert len(api_msgs) == 2
        assert api_msgs[0] == {"role": "user", "content": "Hello"}
        assert api_msgs[1] == {"role": "assistant", "content": "Hi there"}


class TestChatToolCallAccumulation:
    """Test streaming tool_call chunk accumulation (Python equivalent of the JS)."""

    @staticmethod
    def accumulate_tool_calls(deltas):
        """Replicate the toolCallsMap accumulation logic from streamResponse."""
        tool_calls_map = {}
        for delta in deltas:
            if not delta.get("tool_calls"):
                continue
            for tc in delta["tool_calls"]:
                i = tc.get("index", 0)
                if i not in tool_calls_map:
                    tool_calls_map[i] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                if tc.get("id"):
                    tool_calls_map[i]["id"] = tc["id"]
                if tc.get("function", {}).get("name"):
                    tool_calls_map[i]["function"]["name"] += tc["function"]["name"]
                if tc.get("function", {}).get("arguments"):
                    tool_calls_map[i]["function"]["arguments"] += tc["function"]["arguments"]
        return list(tool_calls_map.values())

    def test_single_tool_call(self):
        """A single tool call split across multiple chunks is assembled correctly."""
        deltas = [
            {"tool_calls": [{"index": 0, "id": "tc_1", "function": {"name": "tavily__tavily_search"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"que'}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": 'ry":"test"}'}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "tavily__tavily_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "test"}

    def test_multiple_parallel_tool_calls(self):
        """Multiple tool calls with different indices are accumulated separately."""
        deltas = [
            {"tool_calls": [{"index": 0, "id": "tc_1", "function": {"name": "search"}}]},
            {"tool_calls": [{"index": 1, "id": "tc_2", "function": {"name": "extract"}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"a"}'}}]},
            {"tool_calls": [{"index": 1, "function": {"arguments": '{"urls":["http://x"]}'}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "search"
        assert result[1]["function"]["name"] == "extract"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "a"}
        assert json.loads(result[1]["function"]["arguments"]) == {"urls": ["http://x"]}

    def test_no_tool_calls(self):
        """Deltas with no tool_calls produce empty list."""
        deltas = [
            {"content": "Hello"},
            {"content": " world"},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert result == []

    def test_missing_index_defaults_to_zero(self):
        """A tool_call chunk without an index field defaults to index 0."""
        deltas = [
            {"tool_calls": [{"id": "tc_1", "function": {"name": "t", "arguments": "{}"}}]},
        ]
        result = self.accumulate_tool_calls(deltas)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"


class TestChatToolCallSafety:
    """Test safety guards for the chat tool loop (round limit, abort, errors)."""

    MAX_TOOL_ROUNDS = 10
    TOOL_TIMEOUT_MS = 30000

    @staticmethod
    def build_round_error_message(max_rounds):
        """Replicate the round-limit error message from streamResponse."""
        return (
            f"Error: Maximum tool call rounds ({max_rounds}) reached. "
            "Increase the limit in Chat settings for longer tool workflows."
        )

    @staticmethod
    def normalize_max_tool_rounds(value):
        """Replicate normalizeMaxToolRounds from chat.html."""
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 10
        return min(100, max(1, parsed))

    @staticmethod
    def should_execute_tool_round(completed_rounds, max_rounds):
        """A newly requested tool round is blocked once the limit is reached."""
        return completed_rounds < max_rounds

    @staticmethod
    def build_tool_result(content, error=False, tool_name=None):
        """Replicate the tool execution result format from streamResponse."""
        result = {"content": content, "error": error}
        if tool_name:
            result["toolName"] = tool_name
        return result

    @staticmethod
    def build_timeout_error_message(timeout_ms):
        """Replicate the timeout error message from streamResponse."""
        return f"Error: Tool timed out after {timeout_ms / 1000}s"

    @staticmethod
    def build_tool_status_error(failed_results):
        """Replicate the toolStatus error format from streamResponse."""
        names = [r["toolName"] for r in failed_results if r.get("error")]
        return f"Failed: {', '.join(names)}" if names else ""

    # --- Tool round limit tests ---

    def test_round_limit_error_message_format(self):
        """The error identifies the configured limit and where to change it."""
        msg = self.build_round_error_message(self.MAX_TOOL_ROUNDS)
        assert "10" in msg
        assert "Chat settings" in msg

    def test_tool_round_below_limit_is_executed(self):
        assert self.should_execute_tool_round(9, self.MAX_TOOL_ROUNDS)

    def test_new_tool_round_at_limit_is_blocked(self):
        assert not self.should_execute_tool_round(10, self.MAX_TOOL_ROUNDS)

    def test_custom_tool_round_limit_is_honored(self):
        assert self.should_execute_tool_round(24, 25)
        assert not self.should_execute_tool_round(25, 25)

    def test_tool_round_limit_is_normalized_to_safe_range(self):
        assert self.normalize_max_tool_rounds(None) == 10
        assert self.normalize_max_tool_rounds("bad") == 10
        assert self.normalize_max_tool_rounds(0) == 1
        assert self.normalize_max_tool_rounds(150) == 100

    # --- Tool result format tests ---

    def test_success_result_includes_tool_name(self):
        """Successful tool results should have error=False and include toolName."""
        result = self.build_tool_result("search results here", tool_name="tavily_search")
        assert result["error"] is False
        assert result["toolName"] == "tavily_search"

    def test_error_result_includes_tool_name(self):
        """Failed tool results should have error=True and include toolName."""
        result = self.build_tool_result("Error: connection refused", error=True, tool_name="tavily_search")
        assert result["error"] is True
        assert result["toolName"] == "tavily_search"
        assert result["content"].startswith("Error:")

    def test_timeout_error_message_includes_seconds(self):
        """Timeout error message should show the timeout in seconds."""
        msg = self.build_timeout_error_message(self.TOOL_TIMEOUT_MS)
        assert "30.0s" in msg

    def test_http_error_result_format(self):
        """HTTP errors from /v1/mcp/execute should produce error results."""
        result = self.build_tool_result("Error: HTTP 503", error=True, tool_name="broken_tool")
        assert result["error"] is True
        assert "503" in result["content"]

    # --- Error indicator tests ---

    def test_tool_status_error_format(self):
        """Tool status error message should list failed tool names."""
        failed_results = [
            {"content": "Error: timeout", "error": True, "toolName": "tavily_search"},
            {"content": "Error: HTTP 503", "error": True, "toolName": "broken_tool"},
        ]
        status = self.build_tool_status_error(failed_results)
        assert "tavily_search" in status
        assert "broken_tool" in status
        assert status.startswith("Failed:")

    def test_error_indicators_excluded_from_api(self):
        """Error indicators (role=tool_call) must be filtered from messagesForApi."""
        messages = [
            {"role": "user", "content": "search for X"},
            {"role": "tool_call", "content": "search failed", "_error": True, "_ui": True},
            {"role": "assistant", "content": "Sorry, the search failed."},
        ]
        valid_roles = {"user", "assistant", "tool", "system"}
        api_msgs = [m for m in messages if m["role"] in valid_roles]
        assert len(api_msgs) == 2
        assert all(m["role"] != "tool_call" for m in api_msgs)

    # --- Abort guard tests ---

    def test_abort_signal_prevents_recursion(self):
        """Simulates the abort guard: if signal is aborted, no recursion should happen."""
        # Replicate the guard logic: if (this.abortController?.signal.aborted) return;
        class FakeSignal:
            def __init__(self, aborted):
                self.aborted = aborted

        class FakeController:
            def __init__(self, aborted):
                self.signal = FakeSignal(aborted)

        # When aborted, the guard should fire
        controller = FakeController(aborted=True)
        should_recurse = not (controller.signal.aborted)
        assert should_recurse is False

        # When not aborted, recursion should proceed
        controller = FakeController(aborted=False)
        should_recurse = not (controller.signal.aborted)
        assert should_recurse is True

    def test_abort_guard_with_none_controller(self):
        """If abortController is None, the guard should not crash (optional chaining)."""
        controller = None
        # Replicate JS: this.abortController?.signal.aborted
        aborted = getattr(getattr(controller, "signal", None), "aborted", None)
        # None is falsy, so recursion should proceed
        assert not aborted


class TestChatToolRoundSourceContract:
    """Pin the browser implementation's tool-round and timing lifecycle."""

    @staticmethod
    def stream_response_source():
        source = CHAT_TEMPLATE.read_text(encoding="utf-8")
        start = source.index("async streamResponse(streamContext = null, depth = 0)")
        end = source.index("    stopStreaming()", start)
        return source[start:end]

    def test_limit_is_checked_before_executing_an_extra_tool_round(self):
        stream = self.stream_response_source()
        tool_branch = stream[stream.index("if (toolCalls.length > 0) {") :]

        assert tool_branch.index("if (depth >= maxToolRounds)") < tool_branch.index(
            "const results = await Promise.all"
        )
        assert "MAX_TOOL_DEPTH" not in stream

    def test_final_answer_is_still_allowed_after_the_last_tool_round(self):
        stream = self.stream_response_source()

        assert stream.index("if (toolCalls.length > 0) {") < stream.index(
            "if (depth >= maxToolRounds)"
        )

    def test_root_request_owns_timing_and_stream_cleanup(self):
        stream = self.stream_response_source()

        assert "context._requestStartedAt = Date.now();" in stream
        assert "Date.now() - context._requestStartedAt" in stream
        assert stream.count("this.resetStreamSession(stream") == 2
        finally_body = stream[stream.rindex("} finally {") :]
        assert finally_body.index("if (depth === 0) {") < finally_body.index(
            "this.resetStreamSession(stream, { preserveFinalContent: true });"
        )

    def test_chat_setting_exposes_a_bounded_tool_round_limit(self):
        source = CHAT_TEMPLATE.read_text(encoding="utf-8")

        assert "maxToolRounds: 10" in source
        assert 'id="max-tool-rounds"' in source
        assert 'min="1" max="100"' in source
        assert "normalizeMaxToolRounds(value)" in source


class TestBuiltinWebToolDispatch:
    """Python equivalent of the built-in web tool gating added to chat.html.

    Mirrors webSearchReady / webSearchToolsActive / builtinWebRoute /
    activeTools so the JS contract stays pinned by tests.
    """

    ROUTES = {"web_search": "/v1/web/search", "fetch_url": "/v1/web/fetch"}

    @staticmethod
    def web_search_ready(settings):
        provider = settings.get("provider", "ddgs")
        if provider == "brave":
            return settings.get("braveKeySet", False)
        if provider == "searxng":
            return settings.get("searxngUrlSet", False)
        if provider == "ddgs_custom":
            return settings.get("ddgsBackendsSet", False)
        return True

    def tools_active(self, enabled, settings):
        return enabled and self.web_search_ready(settings)

    def builtin_web_route(self, name, enabled, settings):
        if not self.tools_active(enabled, settings):
            return None
        return self.ROUTES.get(name)

    def active_tools(self, enabled, settings, builtin_tools, mcp_tools):
        if not self.tools_active(enabled, settings):
            return mcp_tools
        builtin_names = set(self.ROUTES)
        return builtin_tools + [
            t for t in mcp_tools if t["function"]["name"] not in builtin_names
        ]

    @staticmethod
    def _tool(name):
        return {"type": "function", "function": {"name": name}}

    def test_toggle_off_keeps_mcp_tools_and_routing(self):
        settings = {"provider": "duckduckgo"}
        mcp_tools = [self._tool("web_search"), self._tool("other")]
        assert self.active_tools(False, settings, [self._tool("web_search")], mcp_tools) == mcp_tools
        # An MCP tool named web_search keeps going to /v1/mcp/execute
        assert self.builtin_web_route("web_search", False, settings) is None

    def test_toggle_on_builtin_wins_name_collision(self):
        settings = {"provider": "duckduckgo"}
        builtin = [self._tool("web_search"), self._tool("fetch_url")]
        mcp_tools = [self._tool("web_search"), self._tool("other")]
        tools = self.active_tools(True, settings, builtin, mcp_tools)
        names = [t["function"]["name"] for t in tools]
        assert names == ["web_search", "fetch_url", "other"]

    def test_builtin_routes_when_active(self):
        settings = {"provider": "duckduckgo"}
        assert self.builtin_web_route("web_search", True, settings) == "/v1/web/search"
        assert self.builtin_web_route("fetch_url", True, settings) == "/v1/web/fetch"
        assert self.builtin_web_route("other", True, settings) is None

    def test_brave_without_key_is_inactive(self):
        settings = {"provider": "brave", "braveKeySet": False}
        assert self.tools_active(True, settings) is False
        settings["braveKeySet"] = True
        assert self.tools_active(True, settings) is True

    def test_searxng_without_url_is_inactive(self):
        settings = {"provider": "searxng", "searxngUrlSet": False}
        assert self.tools_active(True, settings) is False
        settings["searxngUrlSet"] = True
        assert self.tools_active(True, settings) is True

    def test_ddgs_total_and_duckduckgo_need_no_config(self):
        assert self.tools_active(True, {"provider": "ddgs"}) is True
        assert self.tools_active(True, {"provider": "duckduckgo"}) is True

    def test_ddgs_custom_needs_backend_selection(self):
        settings = {"provider": "ddgs_custom", "ddgsBackendsSet": False}
        assert self.tools_active(True, settings) is False
        settings["ddgsBackendsSet"] = True
        assert self.tools_active(True, settings) is True


class TestToolRoundSegmentChain:
    """Mirror of splitTurnSegments/getActiveVariantChain from chat.html.

    Regression guard for the tool-round visibility bug: intermediate
    assistant tool_calls turns must stay _ui:false. A visible assistant
    message is a variant segment boundary, so a visible tool-round turn
    splits the turn and the active chain sent to the API loses the
    tool_calls turn — the orphan tool message then corrupts chat
    templates (observed as garbage output on DeepSeek-V4).
    """

    @staticmethod
    def split_turn_segments(turn):
        segments, current = [], []
        for m in turn:
            current.append(m)
            if m["role"] == "assistant" and m.get("_ui") is not False:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        return segments or [turn]

    def active_chain(self, turn, active_id=None):
        segments = self.split_turn_segments(turn)
        if len(segments) <= 1:
            return turn
        if active_id:
            for segment in segments:
                if any(m.get("id") == active_id for m in segment):
                    return segment
        return segments[-1]

    def test_hidden_tool_round_keeps_full_chain(self):
        turn = [
            {"id": "t1", "role": "assistant", "tool_calls": [{}],
             "_ui": False, "_toolRound": True},
            {"id": "r1", "role": "tool", "_ui": False},
            {"id": "a1", "role": "assistant", "content": "final"},
        ]
        chain = self.active_chain(turn, active_id="a1")
        assert [m["id"] for m in chain] == ["t1", "r1", "a1"]

    def test_visible_tool_round_drops_tool_calls_turn(self):
        # Documents the failure mode this guard exists for.
        turn = [
            {"id": "t1", "role": "assistant", "tool_calls": [{}],
             "_toolRound": True},
            {"id": "r1", "role": "tool", "_ui": False},
            {"id": "a1", "role": "assistant", "content": "final"},
        ]
        chain = self.active_chain(turn, active_id="a1")
        assert [m["id"] for m in chain] == ["r1", "a1"]

    def test_mid_loop_last_segment_is_complete_without_final_answer(self):
        # During recursion (final answer not pushed yet) the last segment
        # must still contain the tool_calls turn and its result.
        turn = [
            {"id": "t1", "role": "assistant", "tool_calls": [{}],
             "_ui": False, "_toolRound": True},
            {"id": "r1", "role": "tool", "_ui": False},
        ]
        chain = self.active_chain(turn)
        assert [m["id"] for m in chain] == ["t1", "r1"]
