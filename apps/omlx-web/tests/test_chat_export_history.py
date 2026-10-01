# SPDX-License-Identifier: Apache-2.0
"""Exercise the chat page's "Download chats" export (#4060)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx_web" / "templates" / "chat.html"

METHODS = [
    "saveCurrentChat",
    "saveChatHistory",
    "sortChatHistory",
    "cloneData",
    "downloadChats",
]


def _sources():
    source = CHAT_TEMPLATE.read_text()
    consts = [
        re.search(r"^ *const " + name + r" = [^\n]*;$", source, re.M).group()
        for name in ("MAX_CHAT_HISTORY_SIZE", "CHAT_HISTORY_STORAGE_KEY")
    ]
    methods = [
        re.search(
            r"^( +)(?:async )?" + name + r"\(.*?\) \{.*?^\1\},",
            source,
            re.M | re.S,
        ).group()
        for name in METHODS
    ]
    return "\n".join(consts), "\n".join(methods)


def _run(body: str):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise Chat JavaScript")
    consts, methods = _sources()
    script = (
        consts
        + """
// localStorage with a fixed quota, like the browser's per-origin limit.
const store = new Map();
const localStorage = {
    quota: Infinity,
    getItem: key => (store.has(key) ? store.get(key) : null),
    setItem(key, value) {
        if (String(value).length > this.quota) {
            const err = new Error('The quota has been exceeded.');
            err.name = 'QuotaExceededError';
            throw err;
        }
        store.set(key, String(value));
    },
    removeItem: key => store.delete(key),
};
const alerts = [];
const alert = message => alerts.push(message);
const window = {t: key => key};
let exported = null;
const URL = {createObjectURL: blob => { exported = blob; return 'blob:chats'; },
             revokeObjectURL() {}};
const document = {
    createElement: () => ({click() {}}),
    body: {appendChild() {}, removeChild() {}},
};
console.error = () => {};
console.warn = () => {};
"""
        + "const app = {"
        + methods
        + "};\n"
        + """
app.chatHistory = [];
app.chatSessions = {};
app.currentChatId = 'chat_1';
app.currentModel = 'model';
app.systemPrompt = '';
app.activePromptProfile = null;
app.getChatSession = id => app.chatSessions[id] || null;
app.resolveAutoChatTitle = (messages, existing) => existing?.title || 'Untitled';
app.syncSessionModelSettingsFromUi = () => {};
app.captureSessionModelSettings = () => ({});
app.saveModelSettingsForModel = () => {};
const exportedChats = async () => {
    exported = null;
    await app.downloadChats();
    return exported ? JSON.parse(await exported.text()) : null;
};
(async () => {
"""
        + body
        + "\n})();\n"
    )
    return json.loads(subprocess.check_output([node, "-e", script], text=True))


def test_export_keeps_messages_that_exceed_the_storage_quota():
    """Messages shown on screen must reach the export even when the browser
    refuses to store them, which left only the system prompt in the file."""
    result = _run("""
const systemPrompt = 'context '.repeat(1000);
const session = {messages: [], model: 'model', systemPrompt, modelSettingsByModel: {}};
app.chatSessions.chat_1 = session;
app.systemPrompt = systemPrompt;
app.saveCurrentChat('chat_1');
// Each assistant reply stores another copy of the system prompt in its meta,
// so the first reply pushes the chat past the quota.
localStorage.quota = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY).length + 100;
session.messages.push({id: 'u1', role: 'user', content: 'question'});
session.messages.push({id: 'a1', role: 'assistant', content: 'answer',
                       meta: {systemPrompt}});
app.saveCurrentChat('chat_1', session.messages, session.model, systemPrompt);
const stored = JSON.parse(localStorage.getItem(CHAT_HISTORY_STORAGE_KEY));
const chats = await exportedChats();
console.log(JSON.stringify({
    storedMessages: stored[0].messages.length,
    exported: chats.map(c => ({id: c.id, systemPrompt: c.systemPrompt === systemPrompt,
                               messages: c.messages.map(m => [m.role, m.content])})),
}));
""")
    # The browser kept the empty snapshot, as in the report.
    assert result["storedMessages"] == 0
    assert result["exported"] == [
        {
            "id": "chat_1",
            "systemPrompt": True,
            "messages": [["user", "question"], ["assistant", "answer"]],
        }
    ]


def test_export_matches_stored_history_and_strips_image_data():
    result = _run("""
const session = {messages: [
    {id: 'u1', role: 'user', content: [
        {type: 'image_url', image_url: {url: 'data:image/png;base64,AAAA'}},
        {type: 'text', text: 'describe'},
    ]},
    {id: 'a1', role: 'assistant', content: 'a cat'},
], model: 'model', systemPrompt: 'be brief', modelSettingsByModel: {}};
app.chatSessions.chat_1 = session;
app.saveCurrentChat('chat_1');
const chats = await exportedChats();
console.log(JSON.stringify({
    sameAsStored: JSON.stringify(chats)
        === JSON.stringify(JSON.parse(localStorage.getItem(CHAT_HISTORY_STORAGE_KEY))),
    image: chats[0].messages[0].content[0],
    alerts,
}));
""")
    assert result["sameAsStored"] is True
    assert result["image"] == {"type": "image_url", "image_url": {"url": ""}}
    assert result["alerts"] == []


def test_export_without_chats_reports_nothing_to_export():
    result = _run("""
const chats = await exportedChats();
console.log(JSON.stringify({chats, alerts}));
""")
    assert result == {"chats": None, "alerts": ["chat.no_chats_to_export"]}
