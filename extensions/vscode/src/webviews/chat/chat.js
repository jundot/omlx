// @ts-check
(function () {
  const vscode = acquireVsCodeApi();

  /** @type {Array<{role: string, content: string}>} */
  let history = [];
  let isStreaming = false;

  const messagesEl = /** @type {HTMLElement} */ (document.getElementById('messages'));
  const emptyStateEl = /** @type {HTMLElement} */ (document.getElementById('emptyState'));
  const inputBox = /** @type {HTMLTextAreaElement} */ (document.getElementById('inputBox'));
  const sendBtn = /** @type {HTMLButtonElement} */ (document.getElementById('sendBtn'));
  const cancelBtn = /** @type {HTMLButtonElement} */ (document.getElementById('cancelBtn'));
  const clearBtn = /** @type {HTMLButtonElement} */ (document.getElementById('clearBtn'));
  const settingsBtn = /** @type {HTMLButtonElement} */ (document.getElementById('settingsBtn'));
  const modelBadge = /** @type {HTMLElement} */ (document.getElementById('modelBadge'));
  const errorBanner = /** @type {HTMLElement} */ (document.getElementById('errorBanner'));
  const tpsInfo = /** @type {HTMLElement} */ (document.getElementById('tpsInfo'));

  // ── Minimal markdown renderer ────────────────────────────
  function renderMarkdown(text) {
    // Escape HTML first
    let html = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Fenced code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
      return `<pre><code class="language-${lang}">${code.trimEnd()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headings
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Unordered lists
    html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`);

    // Paragraphs (double newline separated blocks not already in tags)
    html = html.replace(/\n\n(?!<)/g, '</p><p>');
    if (!html.startsWith('<')) {
      html = '<p>' + html + '</p>';
    }

    return html;
  }

  // ── DOM helpers ──────────────────────────────────────────
  function showEmpty(show) {
    emptyStateEl.style.display = show ? '' : 'none';
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  /**
   * @param {'user'|'assistant'} role
   * @param {string} content
   * @param {boolean} [isMarkdown]
   * @returns {HTMLElement} the message-body element
   */
  function appendMessage(role, content, isMarkdown) {
    showEmpty(false);
    const wrap = document.createElement('div');
    wrap.className = `message ${role}`;

    const roleEl = document.createElement('div');
    roleEl.className = 'message-role';
    roleEl.textContent = role === 'user' ? 'You' : 'oMLX';

    const body = document.createElement('div');
    body.className = 'message-body';
    if (isMarkdown) {
      body.innerHTML = renderMarkdown(content);
    } else {
      body.textContent = content;
    }

    wrap.appendChild(roleEl);
    wrap.appendChild(body);
    messagesEl.appendChild(wrap);
    scrollToBottom();
    return body;
  }

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
    setTimeout(() => errorBanner.classList.add('hidden'), 6000);
  }

  function setStreaming(streaming) {
    isStreaming = streaming;
    sendBtn.classList.toggle('hidden', streaming);
    cancelBtn.classList.toggle('hidden', !streaming);
    inputBox.disabled = streaming;
  }

  // ── Send message ─────────────────────────────────────────
  function sendMessage() {
    const content = inputBox.value.trim();
    if (!content || isStreaming) return;

    inputBox.value = '';
    autoResize();
    errorBanner.classList.add('hidden');

    history.push({ role: 'user', content });
    appendMessage('user', content);

    setStreaming(true);

    let fullContent = '';
    let activeWrap = /** @type {HTMLElement|null} */ (null);
    let activeBody = /** @type {HTMLElement|null} */ (null);
    let activeCursor = /** @type {HTMLElement|null} */ (null);

    /** Creates an assistant bubble on first content, returns it on subsequent calls */
    function ensureBubble() {
      if (activeBody) return;
      showEmpty(false);
      activeWrap = document.createElement('div');
      activeWrap.className = 'message assistant';
      const roleEl = document.createElement('div');
      roleEl.className = 'message-role';
      roleEl.textContent = 'oMLX';
      activeBody = document.createElement('div');
      activeBody.className = 'message-body';
      activeCursor = document.createElement('span');
      activeCursor.className = 'cursor';
      activeBody.appendChild(activeCursor);
      activeWrap.appendChild(roleEl);
      activeWrap.appendChild(activeBody);
      messagesEl.appendChild(activeWrap);
      scrollToBottom();
    }

    vscode.postMessage({ type: 'send-message', content, history: [...history] });

    currentStreamBody = null;
    currentStreamCursor = null;

    /** @param {string} c @param {string} [r] */
    window._onStreamChunk = (c, r) => {
      if (!c && !r) return;
      ensureBubble();
      if (c) {
        fullContent += c;
        activeBody.innerHTML = renderMarkdown(fullContent);
        activeBody.appendChild(activeCursor);
        scrollToBottom();
      }
    };

    window._onToolLoopContinue = () => {
      // Finalize current bubble (if any) and reset for next model turn
      if (activeCursor) activeCursor.remove();
      if (activeBody && fullContent) {
        activeBody.innerHTML = renderMarkdown(fullContent);
      }
      // Reset — next chunk will create a new bubble
      activeWrap = null;
      activeBody = null;
      activeCursor = null;
      fullContent = '';
    };

    window._onStreamDone = (tps) => {
      if (activeCursor) activeCursor.remove();
      if (activeBody) activeBody.innerHTML = renderMarkdown(fullContent);

      if (fullContent) {
        history.push({ role: 'assistant', content: fullContent });

        const codeMatch = fullContent.match(/```[\w]*\n([\s\S]+?)```/);
        if (codeMatch && activeWrap) {
          const insertBtn = document.createElement('button');
          insertBtn.className = 'insert-btn';
          insertBtn.textContent = 'Insert code into editor';
          insertBtn.addEventListener('click', () => {
            vscode.postMessage({ type: 'insert-code', code: codeMatch[1] });
          });
          activeWrap.appendChild(insertBtn);
        }
      }

      if (tps) {
        tpsInfo.textContent = `${tps.toFixed(1)} tok/s`;
        tpsInfo.classList.remove('hidden');
        setTimeout(() => tpsInfo.classList.add('hidden'), 8000);
      }

      setStreaming(false);
      scrollToBottom();
    };

    window._onStreamError = (msg) => {
      if (activeCursor) activeCursor.remove();
      if (activeBody) activeBody.innerHTML = '';
      setStreaming(false);
      showError(`Error: ${msg}`);
    };
  }

  // Placeholders updated by sendMessage
  let currentStreamBody = null;
  let currentStreamCursor = null;

  // ── Textarea auto-resize ─────────────────────────────────
  function autoResize() {
    inputBox.style.height = 'auto';
    inputBox.style.height = Math.min(inputBox.scrollHeight, 120) + 'px';
  }

  // ── Event listeners ──────────────────────────────────────
  sendBtn.addEventListener('click', sendMessage);

  cancelBtn.addEventListener('click', () => {
    vscode.postMessage({ type: 'cancel-stream' });
    setStreaming(false);
    if (currentStreamBody && currentStreamCursor) {
      currentStreamCursor.remove();
    }
  });

  clearBtn.addEventListener('click', () => {
    history = [];
    messagesEl.innerHTML = '';
    messagesEl.appendChild(emptyStateEl);
    showEmpty(true);
    tpsInfo.classList.add('hidden');
    vscode.postMessage({ type: 'clear-history' });
  });

  settingsBtn.addEventListener('click', () => {
    vscode.postMessage({ type: 'open-settings' });
  });

  inputBox.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  inputBox.addEventListener('input', autoResize);

  // ── Tool call block helpers ──────────────────────────────
  /** @type {Map<string, {header: HTMLElement, body: HTMLElement, statusEl: HTMLElement}>} */
  const toolBlockMap = new Map();

  const TOOL_ICONS = {
    execute_code: '⚡',
    run_command: '💻',
    read_file: '📖',
    write_file: '✏️',
    list_files: '📁',
    open_preview: '🌐',
    start_server: '🚀',
  };

  /**
   * @param {string} id
   * @param {string} name
   * @param {Record<string, unknown>} args
   */
  function createToolBlock(id, name, args) {
    showEmpty(false);

    const block = document.createElement('div');
    block.className = 'tool-block';

    const header = document.createElement('div');
    header.className = 'tool-header';

    const icon = document.createElement('span');
    icon.className = 'tool-icon';
    icon.textContent = TOOL_ICONS[name] || '🔧';

    const nameEl = document.createElement('span');
    nameEl.className = 'tool-name';
    nameEl.textContent = name;

    const statusEl = document.createElement('span');
    statusEl.className = 'tool-status running';
    statusEl.textContent = 'running…';

    header.appendChild(icon);
    header.appendChild(nameEl);
    header.appendChild(statusEl);

    const body = document.createElement('div');
    body.className = 'tool-body hidden';

    // Show args in collapsed body
    const argsLabel = document.createElement('div');
    argsLabel.className = 'tool-output-label';
    argsLabel.textContent = 'Arguments';
    const argsContent = document.createElement('pre');
    argsContent.textContent = JSON.stringify(args, null, 2);
    body.appendChild(argsLabel);
    body.appendChild(argsContent);

    // Toggle collapse on click
    header.addEventListener('click', () => {
      body.classList.toggle('hidden');
    });

    block.appendChild(header);
    block.appendChild(body);
    messagesEl.appendChild(block);
    scrollToBottom();

    toolBlockMap.set(id, { header, body, statusEl });
    return { header, body, statusEl };
  }

  /**
   * @param {string} id
   * @param {string} output
   * @param {string|undefined} error
   */
  function updateToolBlock(id, output, error) {
    const refs = toolBlockMap.get(id);
    if (!refs) return;

    const { body, statusEl } = refs;

    statusEl.classList.remove('running');
    if (error) {
      statusEl.classList.add('error');
      statusEl.textContent = 'error';
    } else {
      statusEl.classList.add('done');
      statusEl.textContent = 'done';
    }

    // Append output section
    const outLabel = document.createElement('div');
    outLabel.className = 'tool-output-label';
    outLabel.style.marginTop = '8px';
    outLabel.textContent = error ? 'Error' : 'Output';

    const outContent = document.createElement('pre');
    outContent.textContent = error ? `${error}\n${output}` : output;
    if (error) outContent.classList.add('tool-error');

    body.appendChild(outLabel);
    body.appendChild(outContent);

    // Auto-expand to show result
    body.classList.remove('hidden');
    scrollToBottom();
  }

  // ── Messages from extension host ────────────────────────
  window.addEventListener('message', (event) => {
    const msg = event.data;
    switch (msg.type) {
      case 'stream-chunk':
        if (window._onStreamChunk) {
          window._onStreamChunk(msg.content || '', msg.reasoning || '');
        }
        break;

      case 'stream-done':
        if (window._onStreamDone) {
          window._onStreamDone(msg.usage?.generation_tokens_per_second);
        }
        break;

      case 'stream-error':
        if (window._onStreamError) {
          window._onStreamError(msg.message);
        }
        break;

      case 'tool-start':
        createToolBlock(msg.id, msg.name, msg.args);
        break;

      case 'tool-result':
        updateToolBlock(msg.id, msg.output, msg.error);
        break;

      case 'tool-loop-continue':
        // Model is about to respond again — add a new assistant bubble
        if (window._onToolLoopContinue) {
          window._onToolLoopContinue();
        }
        break;

      case 'model-changed':
        if (msg.model) {
          modelBadge.textContent = msg.model;
          modelBadge.classList.add('loaded');
          emptyStateEl.querySelector('.empty-hint').textContent =
            'Send a message to start';
        } else {
          modelBadge.textContent = 'No model selected';
          modelBadge.classList.remove('loaded');
          emptyStateEl.querySelector('.empty-hint').textContent =
            'Select a model to get started';
        }
        break;
    }
  });

  // Request initial model state
  vscode.postMessage({ type: 'request-models' });
})();
