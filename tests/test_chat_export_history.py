# SPDX-License-Identifier: Apache-2.0
"""Exercise the chat page's "Download chats" export (#4060)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

CHAT_TEMPLATE = Path(__file__).parents[1] / "omlx/admin/templates/chat.html"

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


_TWO_CHATS_THEN_QUOTA = """
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
"""


def _summary(chats):
    return sorted((c["id"], [m["content"] for m in c["messages"]]) for c in chats)


def test_export_keeps_older_chats_when_the_quota_trims_them():
    """Trimming to fit the quota may drop older chats from storage, but the
    export is the backup path and must still contain them."""
    result = _run(_TWO_CHATS_THEN_QUOTA + """
app.chatSessions.chat_1 = newSession();
app.currentChatId = 'chat_1';
say('chat_1', 'first');
localStorage.quota = localStorage.getItem(CHAT_HISTORY_STORAGE_KEY).length + 100;
say('chat_1', 'second');
const chats = await exportedChats();
console.log(JSON.stringify({chats}));
""")
    assert _summary(result["chats"]) == [
        ("chat_1", ["first", "re: first", "second", "re: second"]),
        ("chat_old", ["earlier question", "re: earlier question"]),
    ]


def test_export_keeps_the_current_chat_when_another_chat_is_pinned():
    """Pinned chats sort first, so trimming used to evict the chat in use."""
    result = _run(_TWO_CHATS_THEN_QUOTA + """
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
    assert _summary(result["chats"]) == [
        ("chat_1", ["first", "re: first", "second", "re: second"]),
        ("chat_old", ["earlier question", "re: earlier question"]),
    ]
