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
};
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
    result = _run("""
const systemPrompt = 'context '.repeat(1000);
const session = {messages: [], model: 'model', systemPrompt, modelSettingsByModel: {}};
app.chatSessions.chat_1 = session;
app.systemPrompt = systemPrompt;
app.saveCurrentChat('chat_1');
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
    # Storage still holds the snapshot from before the quota hit.
    assert result["storedMessages"] == 0
    assert result["exported"] == [
        {
            "id": "chat_1",
            "systemPrompt": True,
            "messages": [["user", "question"], ["assistant", "answer"]],
        }
    ]


def test_export_keeps_the_current_chat_when_another_chat_is_pinned():
    """Pinned chats sort first, so the quota trim drops the chat in use."""
    result = _run("""
const systemPrompt = 'context '.repeat(1000);
const newSession = () => ({messages: [], model: 'model', systemPrompt, modelSettingsByModel: {}});
const say = (id, text) => {
    const s = app.chatSessions[id];
    s.messages.push({id: id + '_u' + s.messages.length, role: 'user', content: text});
    s.messages.push({id: id + '_a' + s.messages.length, role: 'assistant', content: 're: ' + text,
                     meta: {systemPrompt}});
    app.saveCurrentChat(id, s.messages, s.model, systemPrompt);
};
app.systemPrompt = systemPrompt;
app.chatSessions.chat_old = newSession();
say('chat_old', 'earlier question');
app.chatHistory.find(c => c.id === 'chat_old').pinned = true;
app.sortChatHistory();
app.saveChatHistory();
app.chatSessions.chat_1 = newSession();
app.currentChatId = 'chat_1';
say('chat_1', 'first');
localStorage.quota = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY).length + 100;
say('chat_1', 'second');
const chats = await exportedChats();
console.log(JSON.stringify({chats}));
""")
    assert sorted(
        (c["id"], [m["content"] for m in c["messages"]]) for c in result["chats"]
    ) == [
        ("chat_1", ["first", "re: first", "second", "re: second"]),
        ("chat_old", ["earlier question", "re: earlier question"]),
    ]
