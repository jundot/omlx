"""Regression guards for the chat UI overhaul follow-up."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHAT_TEMPLATE = ROOT / "omlx_web/templates/chat.html"
I18N_DIR = ROOT / "omlx_web/i18n"
TAILWIND_CSS = ROOT / "omlx_web/static/css/tailwind.css"

NEW_I18N_KEYS = {
    "chat.clear_search",
    "chat.no_chats_match",
    "chat.pin_chat",
    "chat.unpin_chat",
    "chat.regenerate_creative",
    "chat.regenerate_with",
    "chat.chat_tab",
    "chat.max_tool_rounds",
    "chat.max_tool_rounds_hint",
    "chat.error.max_tool_rounds",
}


def _template() -> str:
    return CHAT_TEMPLATE.read_text()


def _section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_regeneration_overrides_survive_stream_context():
    stream = _section(
        _template(),
        "async streamResponse(streamContext = null, depth = 0)",
        "stopStreaming()",
    )

    assert "_modelOverride: streamContext?._modelOverride ?? null" in stream
    assert "_generationOverride: streamContext?._generationOverride ?? null" in stream
    assert "context._modelOverride || context.model" in stream
    assert "context._generationOverride" in stream


def test_regeneration_model_override_reaches_the_request():
    html = _template()
    stream = _section(
        html,
        "async streamResponse(streamContext = null, depth = 0)",
        "stopStreaming()",
    )
    regenerate = _section(
        html,
        "async regenerateMessage(index, opts = {})",
        "_copyFallback(text)",
    )

    assert "if (!context._modelOverride)" in stream
    assert "model: this.currentModel" in regenerate
    assert "_modelOverride: opts.model || null" in regenerate
    assert "chatSession.messages, context.model" not in stream
    assert "chatSession.messages, chatSession.model" in stream


def test_wheel_listener_is_registered_only_during_scroll_setup():
    html = _template()
    setup = _section(html, "    setupScrollListener() {", "    async downloadChats()")
    scroll = _section(
        html,
        "    scrollToBottom(force = false) {",
        "    forceScrollToBottom() {",
    )

    assert "addEventListener('wheel'" in setup
    assert "addEventListener('wheel'" not in scroll


def test_chat_history_is_sorted_before_it_is_trimmed():
    save = _section(
        _template(),
        "    saveCurrentChat(",
        "    startRenamingChat(chat)",
    )

    assert save.index("this.sortChatHistory()") < save.index(
        "this.chatHistory.slice(0, MAX_CHAT_HISTORY_SIZE)"
    )
    assert "this.saveChatHistory()" in save


def test_chat_navigation_preserves_the_previous_chat_timestamp():
    html = _template()
    start_new = _section(
        html, "    async startNewChat(options = {})", "    async loadChat(chatId)"
    )
    load = _section(
        html,
        "    async loadChat(chatId)",
        "    saveCurrentChat(",
    )
    save = _section(
        html,
        "    saveCurrentChat(",
        "    startRenamingChat(chat)",
    )

    assert "{ touchUpdatedAt: false }" in start_new
    assert "{ touchUpdatedAt: false }" in load
    assert "options.touchUpdatedAt === false && existingChat?.updatedAt" in save
    assert "? existingChat.updatedAt" in save


def test_lazy_chat_creation_preserves_preconfigured_draft():
    html = _template()
    start_new = _section(
        html, "    async startNewChat(options = {})", "    async loadChat(chatId)"
    )
    send = _section(
        html,
        "    async sendMessage()",
        "    async sendTranscriptionMessage()",
    )
    transcription = _section(
        html,
        "    async sendTranscriptionMessage()",
        "    async streamTranscription(",
    )
    microphone = _section(
        html,
        "    async startMicTranscription()",
        "    stopMicTranscription(",
    )
    clear_all = _section(
        html,
        "    async clearAllHistory()",
        "    // Thinking/Reasoning tag processing",
    )

    assert (
        "const preserveDraft = options.preserveDraft === true && !prevChatId"
        in start_new
    )
    assert "? draftSystemPrompt" in start_new
    assert "? draftActiveProfile" in start_new
    assert "if (preserveDraft && this.modelSettingsDirty)" in start_new
    assert "this.syncSessionModelSettingsFromUi(session)" in start_new
    assert "this.loadModelCapabilities(" in start_new
    assert "this.resolveGatewayModelId(this.currentModel)" in start_new
    assert (
        "await this.ensureSessionModelSettings(session, this.currentModel)" in start_new
    )

    lazy_create = "await this.startNewChat({ preserveDraft: true })"
    assert lazy_create in send
    assert lazy_create in transcription
    assert lazy_create in microphone
    assert '<button @click="startNewChat()"' in html
    assert "await this.startNewChat();" in clear_all


def test_new_chat_strings_exist_in_every_locale():
    for locale_path in I18N_DIR.glob("*.json"):
        translations = json.loads(locale_path.read_text())
        missing = NEW_I18N_KEYS - translations.keys()
        assert not missing, f"{locale_path.name} is missing {sorted(missing)}"
        assert "{max}" in translations["chat.error.image_too_large"]


def test_tailwind_contains_new_chat_ui_utilities():
    css = TAILWIND_CSS.read_text()

    assert ".max-h-40{" in css
    assert ".z-\\[200\\]{" in css


def test_empty_thinking_content_is_not_rendered_or_replayed():
    html = _template()
    helper = _section(
        html,
        "            hasVisibleThinking(thinking) {",
        "            snapshotGenerationSettings()",
    )
    message_builder = _section(
        html,
        "            buildMessagesForApi(messages, systemPrompt, opts = {})",
        "            buildChatCompletionBody(messages, context, depth)",
    )
    renderer = _section(
        html,
        "    extractThinking(text) {",
        "    // Efficiently update streaming DOM",
    )
    stream = _section(
        html,
        "async streamResponse(streamContext = null, depth = 0)",
        "stopStreaming()",
    )

    assert "thinking.trim().length > 0" in helper
    assert "this.hasVisibleThinking(thinking) ? thinking : null" in helper
    assert "this.hasVisibleThinking(msg.reasoning_content)" in message_builder
    assert "this.hasVisibleThinking(msg._thinking)" in message_builder
    assert "if (content)" in renderer
    assert "if (!this.hasVisibleThinking(content)) return '';" in renderer
    assert "if (this.hasVisibleThinking(thinkingContent))" in renderer
    assert "&& this.hasVisibleThinking(stream.streamingThinking)" in stream
    assert "reasoning_content: this.hasVisibleThinking(stream.streamingThinking)" in stream
    assert 'x-if="hasVisibleThinking(msg._thinking)"' in html
    assert 'x-show="hasVisibleThinking(currentStream()?.streamingThinking)"' in html


def test_regenerate_with_model_switches_the_chat_model():
    regenerate = _section(
        _template(),
        "async regenerateMessage(index, opts = {})",
        "_copyFallback(text)",
    )

    assert "if (opts.model) await this.selectModel(opts.model);" in regenerate


def test_right_panel_dialog_breakpoint_matches_the_stylesheet():
    source = _template()

    assert "max-width: 1024px" in source
    assert "drawerOverlay: window.innerWidth <= 1024" in source
    assert "this.drawerOverlay = window.innerWidth <= 1024" in source


def test_regeneration_hides_the_turn_until_the_new_variant_lands():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = _template()
    names = [
        "findUserIndexForMessage",
        "getTurnVariantsAt",
        "getTurnVariantsForUser",
        "getActiveVariantForUser",
        "isMessageVisible",
        "isToolFlowVisible",
        "isReplacingVariantsAt",
        "isActiveVariantForUser",
        "getStreamSession",
        "currentStream",
    ]
    methods = [
        re.search(
            r"^( +)" + name + r"\([^\n]*\) \{.*?^\1\},", source, re.M | re.S
        ).group()
        for name in names
    ]
    script = (
        "const app = {" + "\n".join(methods) + "};\n" + """
app.currentChatId = 'c';
app.messages = [
    {id: 'u0', role: 'user'}, {id: 'a0', role: 'assistant'},
    {id: 'u1', role: 'user', _activeVariantId: 'a1'},
    {id: 't1', role: 'tool', _web: true}, {id: 'a1', role: 'assistant'},
    {id: 'a2', role: 'assistant'},
];
const visible = () => app.messages
    .filter((m, i) => app.isMessageVisible(m, i)).map(m => m.id);
const stream = {isStreaming: true, targetMessageId: 'a3', replacingVariantsAt: 2};
app.streamSessions = {c: stream};
const during = visible();
const activeTab = app.isActiveVariantForUser(2, 'a1');
app.messages.push({id: 'a3', role: 'assistant'});
app.messages[2]._activeVariantId = 'a3';
const landed = visible();
stream.isStreaming = false;
console.log(JSON.stringify({during, activeTab, landed, after: visible()}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["during"] == ["u0", "a0", "u1"]
    assert result["activeTab"] is False
    assert result["landed"] == ["u0", "a0", "u1", "a3"]
    assert result["after"] == ["u0", "a0", "u1", "a3"]


def test_profile_picks_where_sampling_values_come_from():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = _template()
    names = [
        "currentModelInfo",
        "thinkingModes",
        "thinkingModeValue",
        "reasoningLevels",
        "samplingSource",
        "snapshotGenerationSettings",
        "samplingFields",
        "samplingMode",
        "selectProfileOption",
    ]
    methods = [
        re.search(
            r"^( +)" + name + r"\([^\n]*\) \{.*?^\1\},", source, re.M | re.S
        ).group()
        for name in names
    ]
    script = (
        "const CUSTOM_PROFILE_OPTION = '__custom__';\n"
        "window = {t: key => key};\n"
        "const app = {" + "\n".join(methods) + "};\n" + """
app.currentModel = 'm';
app.aliasToGateway = {};
app._adminModelsList = [{id: 'm', thinking_modes: ['auto'],
    settings: {temperature: 0.6, top_p: 0.95}}];
app.promptProfiles = [{name: 'System Default', content: ''},
    {name: 'Writer', content: 'Write.', params: {temperature: 1.1}}];
app.modelSettings = {temperature: 0.2, enable_thinking: null};
app.onModelSettingsChange = () => {};
app.selectPromptProfile = name => { app.activePromptProfile = name; };
const send = name => {
    app.activePromptProfile = name;
    return app.snapshotGenerationSettings();
};
const model = send('System Default');
const writer = send('Writer');
const custom = send(null);
app.activePromptProfile = 'Writer';
app.selectProfileOption('__custom__');
console.log(JSON.stringify({model, writer, custom,
    seeded: app.snapshotGenerationSettings(), active: app.activePromptProfile}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result["model"] is None
    assert result["writer"] == {"temperature": 1.1}
    assert result["custom"] == {"temperature": 0.2}
    # Switching to Custom Parameters starts from what the profile showed.
    assert result["seeded"] == {"temperature": 1.1, "top_p": 0.95}
    assert result["active"] is None


def test_a_new_turn_uses_the_panel_model_and_settings():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = _template()
    names = [
        "getTurnVariantsAt",
        "getActiveVariantForUser",
        "resolveApiSystemPrompt",
        "resolveApiModel",
        "resolveApiGenerationOverrides",
    ]
    methods = [
        re.search(
            r"^( +)" + name + r"\([^\n]*\) \{.*?^\1\},", source, re.M | re.S
        ).group()
        for name in names
    ]
    script = (
        "const app = {" + "\n".join(methods) + "};\n" + """
app.resolveGatewayModelId = id => id;
app.snapshotGenerationSettings = () => ({reasoning_effort: 'max'});
const messages = [
    {id: 'u0', role: 'user'},
    {id: 'a0', role: 'assistant', meta: {modelId: 'old', systemPrompt: 'Old.',
        generation: {reasoning_effort: 'low'}}},
    {id: 'u1', role: 'user'},
];
console.log(JSON.stringify({
    model: app.resolveApiModel(messages, 'new', 2),
    generation: app.resolveApiGenerationOverrides(messages, 2),
    prompt: app.resolveApiSystemPrompt(messages, 'New.', 2),
}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result == {
        "model": "new",
        "generation": {"reasoning_effort": "max"},
        "prompt": "New.",
    }


def test_picking_a_reply_restores_its_sampling_profile():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    source = _template()
    names = ["normalizeProfileName", "samplingFields", "applyVariantPanelSnapshot"]
    methods = [
        re.search(
            r"^( +)(?:async )?" + name + r"\([^\n]*\) \{.*?^\1\},",
            source,
            re.M | re.S,
        ).group()
        for name in names
    ]
    script = (
        "window = {t: key => key};\n"
        "const app = {" + "\n".join(methods) + "};\n" + """
app.variantModelMissing = () => false;
app.resolveGatewayModelId = id => id;
app.applyGenerationSnapshot = () => {};
app.syncSessionModelSettingsFromUi = () => {};
app.persistChatSettingsToHistory = () => {};
app.loadModelCapabilities = async () => {};
app.promptProfiles = [{name: 'System Default'}, {name: 'Work', params: {}}];
const pick = async (meta) => {
    app.activePromptProfile = 'Work';
    await app.applyVariantPanelSnapshot({meta}, {});
    return app.activePromptProfile;
};
(async () => {
    console.log(JSON.stringify({
        custom: await pick({generation: {temperature: 0.2}}),
        model: await pick({generation: {reasoning_effort: 'low'}}),
        named: await pick({profileName: 'Work', generation: {temperature: 1}}),
        legacy: await pick({}),
    }));
})();
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result == {
        "custom": None,
        "model": "System Default",
        "named": "Work",
        "legacy": "Work",
    }
