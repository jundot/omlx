# SPDX-License-Identifier: Apache-2.0
"""Tests for parser-emitted tool call conversion."""

import logging

from omlx.api.parser_tool_calls import convert_parser_tool_calls


class TestParserToolCallConversion:
    def test_drops_malformed_parser_tool_call_arguments(self, caplog):
        with caplog.at_level(logging.WARNING, logger="omlx.api.parser_tool_calls"):
            converted = convert_parser_tool_calls(
                [
                    {
                        "id": "call_bad",
                        "name": "read",
                        "arguments": '{"path": "/tmp/file"',
                    }
                ]
            )

        assert converted == []
        assert "Dropping malformed parser tool call 'read'" in caplog.text

    def test_keeps_valid_parser_tool_calls_when_one_is_malformed(self):
        converted = convert_parser_tool_calls(
            [
                {
                    "id": "call_valid",
                    "name": "read",
                    "arguments": '{"path": "/tmp/file"}',
                },
                {
                    "id": "call_bad",
                    "name": "write",
                    "arguments": '{"path":',
                },
            ]
        )

        assert len(converted) == 1
        assert converted[0].id == "call_valid"
        assert converted[0].function.name == "read"
        assert converted[0].function.arguments == '{"path": "/tmp/file"}'
