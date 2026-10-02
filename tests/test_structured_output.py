# SPDX-License-Identifier: Apache-2.0
"""
Tests for structured output integration with server.

These tests cover the _inject_json_instruction function from server.py
which is used for injecting JSON schema instructions into messages.
"""

from omlx.server import _inject_json_instruction


class TestInjectJsonInstruction:
    """Tests for _inject_json_instruction function in server."""

    def test_inject_new_system_message(self):
        """Test injecting instruction when no system message exists."""
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Return JSON only" in result[0]["content"]
        assert result[1]["role"] == "user"

    def test_append_to_existing_system(self):
        """Test appending to existing system message."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = _inject_json_instruction(messages, "Return JSON only")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "You are helpful." in result[0]["content"]
        assert "Return JSON only" in result[0]["content"]

    def test_does_not_modify_original(self):
        """Test that original messages are not modified."""
        original = [{"role": "user", "content": "Hello"}]
        original_content = original[0]["content"]
        result = _inject_json_instruction(original, "Return JSON only")

        # Original should be unchanged
        assert len(original) == 1
        assert original[0]["content"] == original_content
        # Result should have 2 messages
        assert len(result) == 2

    def test_inject_with_multiple_user_messages(self):
        """Test injection with multiple user messages."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"},
        ]
        result = _inject_json_instruction(messages, "Use JSON format")

        assert len(result) == 4  # System added
        assert result[0]["role"] == "system"
        assert "Use JSON format" in result[0]["content"]

    def test_inject_with_system_in_middle(self):
        """Test that mid-conversation system messages are not mutated."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Question"},
        ]
        result = _inject_json_instruction(messages, "JSON only")

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert "JSON only" in result[0]["content"]
        assert result[2]["role"] == "system"
        assert result[2]["content"] == "System prompt"

    def test_inject_empty_instruction(self):
        """Test injection with empty instruction."""
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, "")

        assert len(result) == 2
        assert result[0]["role"] == "system"
        # Content should still be set (may just be newlines)
        assert result[0]["content"] is not None

    def test_inject_multiline_instruction(self):
        """Test injection with multiline instruction."""
        instruction = """Please respond with JSON.
The JSON should have these fields:
- name: string
- age: integer"""
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, instruction)

        assert "name: string" in result[0]["content"]
        assert "age: integer" in result[0]["content"]

    def test_inject_preserves_message_order(self):
        """Test that message order is preserved."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Second"},
        ]
        result = _inject_json_instruction(messages, "JSON please")

        assert len(result) == 4
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "First"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert result[3]["content"] == "Second"

    def test_inject_with_pydantic_style_message(self):
        """Test injection with message objects that have attributes."""

        class MockMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [MockMessage("system", "Base instructions")]
        result = _inject_json_instruction(messages, "Add JSON")

        assert len(result) == 1
        # The modified message should have updated content
        assert "Add JSON" in result[0].content
        assert "Base instructions" in result[0].content

    def test_inject_with_dict_and_pydantic_mixed(self):
        """Test injection handles mixed dict and pydantic-style messages."""

        class MockMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        messages = [
            {"role": "user", "content": "Hello"},
            MockMessage("assistant", "Response"),
        ]
        result = _inject_json_instruction(messages, "JSON format")

        assert len(result) == 3
        assert result[0]["role"] == "system"


class TestInjectJsonInstructionEdgeCases:
    """Edge case tests for _inject_json_instruction."""

    def test_empty_messages_list(self):
        """Test with empty messages list."""
        messages = []
        result = _inject_json_instruction(messages, "JSON instruction")

        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert "JSON instruction" in result[0]["content"]

    def test_system_message_with_none_content(self):
        """Test handling system message with None content."""

        class MockMessage:
            def __init__(self):
                self.role = "system"
                self.content = None

        messages = [MockMessage()]
        result = _inject_json_instruction(messages, "Add this")

        # Should handle None content gracefully
        assert len(result) == 1
        # Content should now include the instruction
        assert "Add this" in result[0].content

    def test_special_characters_in_instruction(self):
        """Test instruction with special characters."""
        instruction = '{"type": "object", "properties": {"name": {"type": "string"}}}'
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, instruction)

        assert '{"type": "object"' in result[0]["content"]

    def test_unicode_in_instruction(self):
        """Test instruction with unicode characters."""
        instruction = "Return JSON with Chinese: \u4e2d\u6587"
        messages = [{"role": "user", "content": "Hello"}]
        result = _inject_json_instruction(messages, instruction)

        assert "\u4e2d\u6587" in result[0]["content"]
