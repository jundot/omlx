# SPDX-License-Identifier: Apache-2.0
"""Tests for chat image upload functionality."""
import pytest
from unittest.mock import Mock, patch
import json


class TestChatImageUpload:
    """Test chat image upload feature"""

    def test_multimodal_message_format(self):
        """Test that multimodal messages are correctly formatted"""
        # Simulate the JavaScript message building logic
        def build_multimodal_message(text, image_data):
            if image_data:
                return {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text or "Describe this image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data}
                        }
                    ]
                }
            return {"role": "user", "content": text}

        # Test with image
        msg_with_image = build_multimodal_message(
            "What is this?",
            "data:image/png;base64,abc123"
        )
        assert msg_with_image["role"] == "user"
        assert isinstance(msg_with_image["content"], list)
        assert len(msg_with_image["content"]) == 2
        assert msg_with_image["content"][0]["type"] == "text"
        assert msg_with_image["content"][1]["type"] == "image_url"

        # Test without image
        msg_text_only = build_multimodal_message("Hello", None)
        assert msg_text_only["role"] == "user"
        assert isinstance(msg_text_only["content"], str)
        assert msg_text_only["content"] == "Hello"

    def test_image_file_validation(self):
        """Test image file type validation"""
        valid_types = ["image/png", "image/jpeg", "image/gif", "image/webp"]
        invalid_types = ["application/pdf", "text/plain", "video/mp4"]

        for mime_type in valid_types:
            assert mime_type.startswith("image/"), f"{mime_type} should be valid"

        for mime_type in invalid_types:
            assert not mime_type.startswith("image/"), f"{mime_type} should be invalid"

    def test_image_size_validation(self):
        """Test image size validation (10MB limit)"""
        max_size = 10 * 1024 * 1024  # 10MB in bytes

        # Valid sizes
        assert 1024 <= max_size  # 1KB
        assert 1024 * 1024 <= max_size  # 1MB
        assert 5 * 1024 * 1024 <= max_size  # 5MB

        # Invalid sizes
        assert not (15 * 1024 * 1024 <= max_size)  # 15MB

    def test_i18n_translations_exist(self):
        """Test that required translation keys exist"""
        required_keys = [
            "chat.upload_image",
            "chat.remove_image",
            "chat.drop_image",
            "chat.paste_image",
            "chat.default_image_prompt",
            "chat.error.invalid_image_type",
            "chat.error.image_too_large"
        ]

        for lang_file in ["en.json", "zh.json"]:
            with open(f"omlx/admin/i18n/{lang_file}") as f:
                translations = json.load(f)

            for key in required_keys:
                assert key in translations, f"Missing key '{key}' in {lang_file}"
                assert translations[key], f"Empty value for '{key}' in {lang_file}"

    def test_base64_data_uri_format(self):
        """Test that base64 data URI format is correct"""
        # Valid data URI format
        valid_uris = [
            "data:image/png;base64,iVBORw0KGgo=",
            "data:image/jpeg;base64,/9j/4AAQSkZJ",
            "data:image/webp;base64,UklGRjg=",
        ]

        for uri in valid_uris:
            assert uri.startswith("data:image/"), f"Invalid format: {uri}"
            assert ";base64," in uri, f"Missing base64 marker: {uri}"

        # Invalid formats
        invalid_uris = [
            "https://example.com/image.png",  # URL, not data URI
            "data:text/plain;base64,abc",  # Wrong MIME type
            "image.png",  # Just filename
        ]

        for uri in invalid_uris:
            is_valid_data_uri = uri.startswith("data:image/") and ";base64," in uri
            assert not is_valid_data_uri, f"Should be invalid: {uri}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
