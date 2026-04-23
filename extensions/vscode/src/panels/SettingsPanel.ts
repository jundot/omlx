// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { OmlxClient } from '../api/client.js';
import type { VscodeConfig } from '../api/types.js';

type WebviewMessage =
  | { type: 'ready' }
  | { type: 'save-vscode-config'; config: Partial<VscodeConfig> }
  | { type: 'load-model'; modelId: string }
  | { type: 'unload-model'; modelId: string }
  | { type: 'refresh' };

export class SettingsPanel {
  public static readonly viewType = 'omlx.settingsPanel';

  private static _current: SettingsPanel | undefined;

  private readonly _panel: vscode.WebviewPanel;
  private _disposables: vscode.Disposable[] = [];

  static createOrShow(extensionUri: vscode.Uri, client: OmlxClient): void {
    if (SettingsPanel._current) {
      SettingsPanel._current._panel.reveal(vscode.ViewColumn.One);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      SettingsPanel.viewType,
      'oMLX Settings',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, 'src', 'webviews', 'settings'),
        ],
      },
    );
    SettingsPanel._current = new SettingsPanel(panel, extensionUri, client);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    private readonly _extensionUri: vscode.Uri,
    private readonly _client: OmlxClient,
  ) {
    this._panel = panel;
    this._panel.webview.html = this._getHtml();
    this._panel.onDidDispose(() => this._dispose(), null, this._disposables);
    this._panel.webview.onDidReceiveMessage(
      (msg: WebviewMessage) => this._handleMessage(msg),
      null,
      this._disposables,
    );
  }

  private async _handleMessage(msg: WebviewMessage): Promise<void> {
    switch (msg.type) {
      case 'ready':
        await this._sendSettings();
        break;

      case 'refresh':
        await this._sendSettings();
        break;

      case 'save-vscode-config':
        await this._saveConfig(msg.config);
        break;

      case 'load-model':
        // Fire the load request and immediately start polling — loading can take 60+ seconds
        this._client.loadModel(msg.modelId).catch(() => {
          // 409 means already loading (fine), other errors surface via poll
        });
        this._pollUntilLoaded(msg.modelId);
        break;

      case 'unload-model':
        try {
          await this._client.unloadModel(msg.modelId);
          this._panel.webview.postMessage({
            type: 'model-load-result',
            modelId: msg.modelId,
            success: true,
          });
        } catch (e) {
          this._panel.webview.postMessage({
            type: 'error',
            message: `Failed to unload ${msg.modelId}: ${e instanceof Error ? e.message : String(e)}`,
          });
        }
        break;
    }
  }

  private _pollUntilLoaded(modelId: string, maxAttempts = 60): void {
    let attempts = 0;
    const interval = setInterval(async () => {
      attempts++;
      try {
        const models = await this._client.listAdminModels();
        const model = models.find((m) => m.id === modelId);
        if (model?.loaded) {
          clearInterval(interval);
          this._panel.webview.postMessage({
            type: 'model-load-result',
            modelId,
            success: true,
          });
          await this._sendSettings();
        } else if (attempts >= maxAttempts) {
          clearInterval(interval);
          this._panel.webview.postMessage({
            type: 'error',
            message: `Timed out waiting for ${modelId} to load`,
          });
        }
      } catch {
        // Server busy — keep polling
      }
    }, 2000);
    this._disposables.push({ dispose: () => clearInterval(interval) });
  }

  private async _sendSettings(): Promise<void> {
    const config = vscode.workspace.getConfiguration('omlx');
    const vscodeConfig: VscodeConfig = {
      serverUrl: config.get<string>('serverUrl', 'http://localhost:8000'),
      apiKey: config.get<string>('apiKey', ''),
      completionsEnabled: config.get<boolean>('completions.enabled', true),
      completionsMaxTokens: config.get<number>('completions.maxTokens', 80),
      completionsDebounceMs: config.get<number>('completions.debounceMs', 300),
      completionsContextLines: config.get<number>('completions.contextLines', 100),
      chatDefaultSystemPrompt: config.get<string>(
        'chat.defaultSystemPrompt',
        'You are a helpful programming assistant.',
      ),
    };

    let serverSettings = null;
    let stats = null;
    let models = null;

    try {
      [serverSettings, stats, models] = await Promise.all([
        this._client.getGlobalSettings(),
        this._client.getStats(),
        this._client.listAdminModels(),
      ]);
    } catch {
      // Server may be offline; send what we have
    }

    this._panel.webview.postMessage({
      type: 'settings-loaded',
      vscodeConfig,
      serverSettings,
      stats,
      models,
    });
  }

  private async _saveConfig(partial: Partial<VscodeConfig>): Promise<void> {
    const config = vscode.workspace.getConfiguration('omlx');
    const target = vscode.ConfigurationTarget.Global;

    if (partial.serverUrl !== undefined) {
      await config.update('serverUrl', partial.serverUrl, target);
    }
    if (partial.apiKey !== undefined) {
      await config.update('apiKey', partial.apiKey, target);
    }
    if (partial.completionsEnabled !== undefined) {
      await config.update('completions.enabled', partial.completionsEnabled, target);
    }
    if (partial.completionsMaxTokens !== undefined) {
      await config.update('completions.maxTokens', partial.completionsMaxTokens, target);
    }
    if (partial.completionsDebounceMs !== undefined) {
      await config.update('completions.debounceMs', partial.completionsDebounceMs, target);
    }
    if (partial.completionsContextLines !== undefined) {
      await config.update('completions.contextLines', partial.completionsContextLines, target);
    }
    if (partial.chatDefaultSystemPrompt !== undefined) {
      await config.update('chat.defaultSystemPrompt', partial.chatDefaultSystemPrompt, target);
    }

    this._panel.webview.postMessage({ type: 'save-success' });
  }

  private _getHtml(): string {
    const webview = this._panel.webview;
    const nonce = getNonce();
    const base = vscode.Uri.joinPath(this._extensionUri, 'src', 'webviews', 'settings');
    const cssUri = webview.asWebviewUri(vscode.Uri.joinPath(base, 'settings.css'));
    const jsUri = webview.asWebviewUri(vscode.Uri.joinPath(base, 'settings.js'));

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}'; font-src ${webview.cspSource};" />
  <link rel="stylesheet" href="${cssUri}" />
  <title>oMLX Settings</title>
</head>
<body>
  <h1>
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
    oMLX Settings
  </h1>

  <div class="error-banner hidden" id="errorBanner"></div>

  <!-- Extension Settings -->
  <section>
    <h2>Extension</h2>

    <div class="field">
      <label for="serverUrl">Server URL</label>
      <input type="text" id="serverUrl" placeholder="http://localhost:8000" />
      <div class="field-desc">URL where oMLX is running</div>
    </div>

    <div class="field">
      <label for="apiKey">API Key</label>
      <input type="password" id="apiKey" placeholder="Leave empty if auth is disabled" />
    </div>

    <div class="field">
      <div class="checkbox-row">
        <input type="checkbox" id="completionsEnabled" />
        <label for="completionsEnabled">Enable inline completions</label>
      </div>
    </div>

    <div class="field">
      <label for="maxTokens">Max completion tokens</label>
      <input type="number" id="maxTokens" min="1" max="2048" />
    </div>

    <div class="field">
      <label for="debounceMs">Completion debounce (ms)</label>
      <input type="number" id="debounceMs" min="50" max="2000" />
      <div class="field-desc">Delay before triggering completions after you stop typing</div>
    </div>

    <div class="field">
      <label for="contextLines">Context lines</label>
      <input type="number" id="contextLines" min="10" max="500" />
      <div class="field-desc">Lines of code before cursor sent as context</div>
    </div>

    <div class="field">
      <label for="systemPrompt">Default system prompt</label>
      <textarea id="systemPrompt" rows="3"></textarea>
    </div>

    <div class="save-row">
      <button class="btn btn-primary" id="saveBtn">Save</button>
      <span class="save-status" id="saveStatus">&#10003; Saved</span>
    </div>
  </section>

  <!-- Server Info -->
  <section>
    <h2>Server info <button class="btn btn-secondary" id="refreshBtn" style="font-size:11px;padding:2px 8px;margin-left:8px;">Refresh</button></h2>
    <div class="info-grid">
      <span class="info-key">Host</span><span id="serverHost">—</span>
      <span class="info-key">Port</span><span id="serverPort">—</span>
      <span class="info-key">Cache</span><span id="serverCache">—</span>
      <span class="info-key">Max memory</span><span id="serverMaxMem">—</span>
    </div>
  </section>

  <!-- Stats -->
  <section>
    <h2>Session stats</h2>
    <div class="stats-grid">
      <div class="stat-card">
        <div class="stat-label">Requests</div>
        <div class="stat-value" id="statRequests">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Tokens generated</div>
        <div class="stat-value" id="statTokens">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg gen speed</div>
        <div class="stat-value" id="statGenTps">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Avg prefill speed</div>
        <div class="stat-value" id="statPrefillTps">—</div>
      </div>
    </div>
  </section>

  <!-- Model Manager -->
  <section>
    <h2>Models</h2>
    <table class="model-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Size</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody id="modelTableBody">
        <tr><td colspan="3" style="color:var(--vscode-descriptionForeground);padding:12px 8px;">Loading…</td></tr>
      </tbody>
    </table>
  </section>

  <script nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
  }

  private _dispose(): void {
    SettingsPanel._current = undefined;
    for (const d of this._disposables) {
      d.dispose();
    }
    this._disposables = [];
  }
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < 32; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
