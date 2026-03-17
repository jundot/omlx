# SPDX-License-Identifier: Apache-2.0
"""Tests for chat image upload functionality."""
import json
from pathlib import Path

import pytest


I18N_DIR = Path(__file__).parent.parent / "omlx" / "admin" / "i18n"

# Required i18n keys used by chat image upload feature
REQUIRED_IMAGE_KEYS = [
    "chat.upload_image",
    "chat.remove_image",
    "chat.image_not_available",
    "chat.image_preview",
    "chat.error.invalid_image_type",
    "chat.error.image_too_large",
    "chat.error.image_load_failed",
]


class TestChatImageUpload:
    """Test chat image upload feature"""

    def test_multimodal_message_format_with_images(self):
        """Content array format for messages with images (OpenAI standard)"""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,def"}},
            {"type": "text", "text": "What are these?"},
        ]
        msg = {"role": "user", "content": content}

        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        images = [p for p in msg["content"] if p["type"] == "image_url"]
        texts = [p for p in msg["content"] if p["type"] == "text"]
        assert len(images) == 2
        assert len(texts) == 1
        assert all(p["image_url"]["url"].startswith("data:image/") for p in images)

    def test_multimodal_message_format_text_only(self):
        """Text-only messages use plain string content"""
        msg = {"role": "user", "content": "Hello"}
        assert isinstance(msg["content"], str)

    def test_multimodal_message_format_image_only(self):
        """Image-only messages have no text part"""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        msg = {"role": "user", "content": content}
        texts = [p for p in msg["content"] if p["type"] == "text"]
        assert len(texts) == 0

    def test_localstorage_stripping(self):
        """Stripping base64 from image_url parts for localStorage"""
        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,LARGE"}},
            {"type": "text", "text": "describe this"},
        ]
        # Simulate the stripping logic from saveCurrentChat()
        stripped = [
            {"type": "image_url", "image_url": {"url": ""}}
            if p["type"] == "image_url"
            else p
            for p in content
        ]
        assert stripped[0]["image_url"]["url"] == ""
        assert stripped[1]["text"] == "describe this"

    def test_base64_data_uri_format(self):
        """Valid base64 data URIs for images"""
        valid_uris = [
            "data:image/png;base64,iVBORw0KGgo=",
            "data:image/jpeg;base64,/9j/4AAQSkZJ",
            "data:image/webp;base64,UklGRjg=",
        ]
        for uri in valid_uris:
            assert uri.startswith("data:image/")
            assert ";base64," in uri

    @pytest.mark.parametrize(
        "lang_file",
        ["en.json", "ko.json", "zh.json", "zh-TW.json", "ja.json"],
    )
    def test_i18n_image_keys_present(self, lang_file):
        """All image-related i18n keys exist in every language file"""
        with open(I18N_DIR / lang_file) as f:
            translations = json.load(f)

        for key in REQUIRED_IMAGE_KEYS:
            assert key in translations, f"Missing key '{key}' in {lang_file}"
            assert translations[key], f"Empty value for '{key}' in {lang_file}"


class TestChatEditImagePreservation:
    """Test image preservation when editing messages (PR #268)"""

    @staticmethod
    def get_image_urls(content):
        """Simulate getImageUrls() from chat.html"""
        if not isinstance(content, list):
            return []
        return [
            p["image_url"]["url"]
            for p in content
            if p.get("type") == "image_url" and p.get("image_url", {}).get("url")
        ]

    @staticmethod
    def get_text_content(content):
        """Simulate getTextContent() from chat.html"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                p["text"] for p in content if p.get("type") == "text"
            )
        return ""

    @staticmethod
    def build_edit_content(edit_images, new_text):
        """Simulate saveEdit() content reconstruction from chat.html"""
        if edit_images:
            content = []
            for img in edit_images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            content.append({"type": "text", "text": new_text})
            return content
        return new_text

    def test_edit_preserves_images(self):
        """Editing a message with images should preserve the images"""
        original = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,def"}},
            {"type": "text", "text": "What are these?"},
        ]
        edit_images = self.get_image_urls(original)
        edit_text = self.get_text_content(original)

        assert edit_images == [
            "data:image/png;base64,abc",
            "data:image/jpeg;base64,def",
        ]
        assert edit_text == "What are these?"

        new_content = self.build_edit_content(edit_images, "Describe these images")
        assert isinstance(new_content, list)
        images = [p for p in new_content if p["type"] == "image_url"]
        texts = [p for p in new_content if p["type"] == "text"]
        assert len(images) == 2
        assert len(texts) == 1
        assert texts[0]["text"] == "Describe these images"

    def test_edit_text_only_message(self):
        """Editing a text-only message should produce string content"""
        original = "Hello"
        edit_images = self.get_image_urls(original)
        edit_text = self.get_text_content(original)

        assert edit_images == []
        assert edit_text == "Hello"

        new_content = self.build_edit_content(edit_images, "Hi there")
        assert isinstance(new_content, str)
        assert new_content == "Hi there"

    def test_edit_stripped_images_not_preserved(self):
        """Stripped images (empty URL from localStorage) should not be preserved"""
        restored = [
            {"type": "image_url", "image_url": {"url": ""}},
            {"type": "text", "text": "old text"},
        ]
        edit_images = self.get_image_urls(restored)
        assert edit_images == []

        new_content = self.build_edit_content(edit_images, "edited text")
        assert isinstance(new_content, str)
        assert new_content == "edited text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
