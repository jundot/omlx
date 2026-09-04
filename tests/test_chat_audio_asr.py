# SPDX-License-Identifier: Apache-2.0
"""Tests for the chat ASR (transcription) UI wiring in chat.html."""

import json
from pathlib import Path

import pytest

I18N_DIR = Path(__file__).parent.parent / "omlx" / "admin" / "i18n"
CHAT_TEMPLATE = (
    Path(__file__).parent.parent / "omlx" / "admin" / "templates" / "chat.html"
)

# Required i18n keys used by the chat ASR feature
REQUIRED_AUDIO_KEYS = [
    "chat.upload_audio",
    "chat.remove_audio",
    "chat.record_audio",
    "chat.stop_recording",
    "chat.mic_recording_label",
    "chat.input_placeholder_asr",
    "chat.status.transcribing",
    "chat.status.listening",
    "chat.error.invalid_audio_type",
    "chat.error.audio_too_large",
    "chat.error.asr_not_available",
    "chat.error.mic_unavailable",
]

ALL_LOCALES = sorted(p.name for p in I18N_DIR.glob("*.json"))


class TestChatAsrTemplate:
    """The chat template contains the ASR UI hooks."""

    @pytest.fixture(scope="class")
    def template(self):
        return CHAT_TEMPLATE.read_text(encoding="utf-8")

    def test_asr_gate_and_send_branch_present(self, template):
        assert "hasAsrSupport()" in template
        assert "sendTranscriptionMessage(" in template
        assert "streamTranscription(" in template

    def test_audio_input_and_handlers_present(self, template):
        assert 'id="audioInput"' in template
        assert "handleAudioSelect(" in template
        assert "loadAudioFile(" in template

    def test_transcriptions_endpoint_used(self, template):
        assert "/v1/audio/transcriptions" in template

    def test_realtime_mic_wiring_present(self, template):
        assert "toggleMicTranscription(" in template
        assert "isRealtimeSttModel()" in template
        assert "/v1/audio/transcriptions/realtime" in template
        assert "realtime_stt" in template

    def test_model_picker_allows_audio_stt(self, template):
        # audio_stt must not be filtered out of the picker; tts/sts stay out
        assert "!t.startsWith('audio_')" not in template
        assert "t !== 'audio_tts' && t !== 'audio_sts'" in template

    def test_transcription_turns_guard_regenerate(self, template):
        assert "origMsg._transcription" in template


class TestChatAsrI18n:
    @pytest.mark.parametrize("lang_file", ALL_LOCALES)
    def test_i18n_audio_keys_present(self, lang_file):
        """All ASR-related i18n keys exist in every language file."""
        with open(I18N_DIR / lang_file, encoding="utf-8") as f:
            translations = json.load(f)

        for key in REQUIRED_AUDIO_KEYS:
            assert key in translations, f"Missing key '{key}' in {lang_file}"
            assert translations[key], f"Empty value for '{key}' in {lang_file}"
