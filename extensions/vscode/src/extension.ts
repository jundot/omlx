// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import { OmlxClient } from './api/client.js';
import { registerCommands } from './commands/index.js';
import { ChatPanel } from './panels/ChatPanel.js';
import { SettingsPanel } from './panels/SettingsPanel.js';
import { OmlxInlineCompletionProvider } from './providers/inlineCompletion.js';
import { StatusBarManager } from './providers/statusBar.js';
import { state } from './state/ExtensionState.js';

const HEALTH_POLL_INTERVAL_MS = 10_000;
let _shownAuthWarning = false;

export function activate(context: vscode.ExtensionContext): void {
  // Initialize state with persistent storage
  state.init(context);

  // Build the client — reads config on every call so no stale credentials
  const client = new OmlxClient(
    () => vscode.workspace.getConfiguration('omlx').get<string>('serverUrl', 'http://localhost:8000'),
    () => vscode.workspace.getConfiguration('omlx').get<string>('apiKey', ''),
  );

  // ── Status bar ─────────────────────────────────────────
  const statusBar = new StatusBarManager(state);
  state.onDidChange(() => statusBar.notifyStateChanged());
  context.subscriptions.push({ dispose: () => statusBar.dispose() });

  // ── Health polling + auto-select model ────────────────
  async function pollHealth(): Promise<void> {
    const healthy = await client.healthCheck();
    state.serverHealthy = healthy;

    if (healthy && !state.selectedModel) {
      try {
        const loaded = await client.listModels();
        if (loaded.length > 0) {
          state.selectedModel = loaded[0].id;
        } else {
          const admin = await client.listAdminModels();
          const pick =
            admin.find((m) => m.is_default && m.loaded) ??
            admin.find((m) => m.loaded) ??
            admin[0];
          if (pick) {
            state.selectedModel = pick.id;
          }
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        if (msg.includes('401') && !_shownAuthWarning) {
          _shownAuthWarning = true;
          vscode.window
            .showWarningMessage(
              'oMLX: API key required. Set it in oMLX Settings.',
              'Open Settings',
            )
            .then((choice) => {
              if (choice === 'Open Settings') {
                vscode.commands.executeCommand('omlx.openSettingsPanel');
              }
            });
        }
      }
    }
  }
  void pollHealth();
  const healthTimer = setInterval(() => void pollHealth(), HEALTH_POLL_INTERVAL_MS);
  context.subscriptions.push({ dispose: () => clearInterval(healthTimer) });

  // ── Inline completions ─────────────────────────────────
  const completionProvider = new OmlxInlineCompletionProvider(client, state);
  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider(
      { pattern: '**' },
      completionProvider,
    ),
    { dispose: () => completionProvider.dispose() },
  );

  // ── Chat sidebar ───────────────────────────────────────
  const chatPanel = new ChatPanel(context.extensionUri, client, state);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatPanel.viewType, chatPanel),
    { dispose: () => chatPanel.dispose() },
  );

  // ── Settings panel command ─────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand('omlx.openSettingsPanel', () => {
      SettingsPanel.createOrShow(context.extensionUri, client);
    }),
  );

  // ── Other commands ─────────────────────────────────────
  registerCommands(context, client, state);

  // ── Config change listener ─────────────────────────────
  // Re-poll health immediately when server URL changes
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration('omlx.serverUrl')) {
        void pollHealth();
      }
    }),
  );
}

export function deactivate(): void {
  // Interval and disposables are cleaned up via context.subscriptions
}
