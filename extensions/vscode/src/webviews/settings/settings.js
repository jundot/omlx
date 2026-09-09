// @ts-check
(function () {
  const vscode = acquireVsCodeApi();

  // ── DOM refs ─────────────────────────────────────────────
  const errorBanner = /** @type {HTMLElement} */ (document.getElementById('errorBanner'));
  const saveStatus = /** @type {HTMLElement} */ (document.getElementById('saveStatus'));

  // Extension config fields
  const serverUrlInput = /** @type {HTMLInputElement} */ (document.getElementById('serverUrl'));
  const apiKeyInput = /** @type {HTMLInputElement} */ (document.getElementById('apiKey'));
  const completionsEnabledInput = /** @type {HTMLInputElement} */ (document.getElementById('completionsEnabled'));
  const maxTokensInput = /** @type {HTMLInputElement} */ (document.getElementById('maxTokens'));
  const debounceMsInput = /** @type {HTMLInputElement} */ (document.getElementById('debounceMs'));
  const contextLinesInput = /** @type {HTMLInputElement} */ (document.getElementById('contextLines'));
  const systemPromptInput = /** @type {HTMLTextAreaElement} */ (document.getElementById('systemPrompt'));

  // Server info
  const serverHostEl = /** @type {HTMLElement} */ (document.getElementById('serverHost'));
  const serverPortEl = /** @type {HTMLElement} */ (document.getElementById('serverPort'));
  const serverCacheEl = /** @type {HTMLElement} */ (document.getElementById('serverCache'));
  const serverMaxMemEl = /** @type {HTMLElement} */ (document.getElementById('serverMaxMem'));

  // Stats
  const statRequestsEl = /** @type {HTMLElement} */ (document.getElementById('statRequests'));
  const statTokensEl = /** @type {HTMLElement} */ (document.getElementById('statTokens'));
  const statGenTpsEl = /** @type {HTMLElement} */ (document.getElementById('statGenTps'));
  const statPrefillTpsEl = /** @type {HTMLElement} */ (document.getElementById('statPrefillTps'));

  // Model table
  const modelTableBody = /** @type {HTMLElement} */ (document.getElementById('modelTableBody'));

  // ── Helpers ──────────────────────────────────────────────
  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.remove('hidden');
  }

  function flashSave() {
    saveStatus.classList.add('visible');
    setTimeout(() => saveStatus.classList.remove('visible'), 2500);
  }

  function populateConfig(config) {
    if (!config) return;
    serverUrlInput.value = config.serverUrl ?? '';
    apiKeyInput.value = config.apiKey ?? '';
    completionsEnabledInput.checked = config.completionsEnabled ?? true;
    maxTokensInput.value = String(config.completionsMaxTokens ?? 80);
    debounceMsInput.value = String(config.completionsDebounceMs ?? 300);
    contextLinesInput.value = String(config.completionsContextLines ?? 100);
    systemPromptInput.value = config.chatDefaultSystemPrompt ?? '';
  }

  function populateServerSettings(settings) {
    if (!settings) return;
    if (settings.server) {
      serverHostEl.textContent = settings.server.host ?? '—';
      serverPortEl.textContent = String(settings.server.port ?? '—');
    }
    if (settings.cache) {
      serverCacheEl.textContent = settings.cache.enabled ? 'Enabled' : 'Disabled';
    }
    if (settings.model) {
      serverMaxMemEl.textContent = settings.model.max_model_memory ?? '—';
    }
  }

  function populateStats(stats) {
    if (!stats) return;
    statRequestsEl.textContent = String(stats.total_requests ?? 0);
    statTokensEl.textContent = String(stats.total_completion_tokens ?? 0);
    statGenTpsEl.textContent = (stats.avg_generation_tps ?? 0).toFixed(1);
    statPrefillTpsEl.textContent = (stats.avg_prefill_tps ?? 0).toFixed(1);
  }

  function populateModels(models) {
    if (!models) return;
    modelTableBody.innerHTML = '';
    for (const m of models) {
      const tr = document.createElement('tr');

      const statusClass = m.loaded ? 'loaded' : m.is_loading ? 'loading' : 'unloaded';
      const statusText = m.loaded ? 'Loaded' : m.is_loading ? 'Loading…' : 'Unloaded';

      tr.innerHTML = `
        <td>
          <span class="status-dot ${statusClass}"></span>${m.id}
          ${m.is_default ? ' <small style="opacity:0.5">(default)</small>' : ''}
        </td>
        <td>${m.estimated_size_formatted ?? '—'}</td>
        <td>
          <div class="model-actions">
            <button data-action="load" data-id="${m.id}" ${m.loaded || m.is_loading ? 'disabled' : ''}>Load</button>
            <button data-action="unload" data-id="${m.id}" ${!m.loaded ? 'disabled' : ''}>Unload</button>
          </div>
        </td>
      `;
      modelTableBody.appendChild(tr);
    }
  }

  // ── Event: save extension config ─────────────────────────
  document.getElementById('saveBtn').addEventListener('click', () => {
    vscode.postMessage({
      type: 'save-vscode-config',
      config: {
        serverUrl: serverUrlInput.value.trim(),
        apiKey: apiKeyInput.value,
        completionsEnabled: completionsEnabledInput.checked,
        completionsMaxTokens: parseInt(maxTokensInput.value, 10) || 80,
        completionsDebounceMs: parseInt(debounceMsInput.value, 10) || 300,
        completionsContextLines: parseInt(contextLinesInput.value, 10) || 100,
        chatDefaultSystemPrompt: systemPromptInput.value,
      },
    });
  });

  // ── Event: model load/unload buttons ────────────────────
  modelTableBody.addEventListener('click', (e) => {
    const btn = /** @type {HTMLButtonElement} */ (e.target);
    if (btn.tagName !== 'BUTTON') return;
    const action = btn.dataset.action;
    const id = btn.dataset.id;
    if (!id) return;

    // Disable all buttons in this row and show spinner on Load
    const row = btn.closest('tr');
    if (row) {
      row.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    }

    if (action === 'load') {
      btn.textContent = 'Loading…';
      vscode.postMessage({ type: 'load-model', modelId: id });
    } else if (action === 'unload') {
      btn.textContent = 'Unloading…';
      vscode.postMessage({ type: 'unload-model', modelId: id });
    }
  });

  // ── Event: refresh ───────────────────────────────────────
  document.getElementById('refreshBtn').addEventListener('click', () => {
    vscode.postMessage({ type: 'refresh' });
  });

  // ── Messages from extension host ────────────────────────
  window.addEventListener('message', (event) => {
    const msg = event.data;

    switch (msg.type) {
      case 'settings-loaded':
        errorBanner.classList.add('hidden');
        populateConfig(msg.vscodeConfig);
        populateServerSettings(msg.serverSettings);
        populateStats(msg.stats);
        populateModels(msg.models);
        break;

      case 'save-success':
        flashSave();
        break;

      case 'model-load-result':
        // Re-request full refresh to update model table
        vscode.postMessage({ type: 'refresh' });
        break;

      case 'error':
        showError(msg.message);
        break;
    }
  });

  // Signal ready — host will send settings-loaded
  vscode.postMessage({ type: 'ready' });
})();
