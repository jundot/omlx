// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { OmlxClient } from '../api/client.js';
import type { ChatMessage, ToolCall } from '../api/types.js';
import type { ExtensionState } from '../state/ExtensionState.js';
import { TOOL_DEFINITIONS } from '../tools/definitions.js';
import { executeTool } from '../tools/executor.js';

const MAX_AGENT_ITERATIONS = 10;

type WebviewMessage =
  | { type: 'send-message'; content: string; history: ChatMessage[] }
  | { type: 'cancel-stream' }
  | { type: 'request-models' }
  | { type: 'clear-history' }
  | { type: 'open-settings' }
  | { type: 'insert-code'; code: string };

export class ChatPanel implements vscode.WebviewViewProvider {
  public static readonly viewType = 'omlx.chatView';

  private _view: vscode.WebviewView | undefined;
  private _abortController: AbortController | undefined;
  private _stateDisposable: vscode.Disposable | undefined;

  constructor(
    private readonly _extensionUri: vscode.Uri,
    private readonly _client: OmlxClient,
    private readonly _state: ExtensionState,
  ) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken,
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this._extensionUri, 'src', 'webviews', 'chat'),
      ],
    };

    webviewView.webview.html = this._getHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage((msg: WebviewMessage) => {
      void this._handleMessage(msg);
    });

    this._stateDisposable = this._state.onDidChange((s) => {
      webviewView.webview.postMessage({ type: 'model-changed', model: s.selectedModel ?? null });
    });

    webviewView.webview.postMessage({ type: 'model-changed', model: this._state.selectedModel ?? null });
  }

  private async _handleMessage(msg: WebviewMessage): Promise<void> {
    switch (msg.type) {
      case 'send-message':
        await this._runAgentLoop(msg.history);
        break;
      case 'cancel-stream':
        this._abortController?.abort();
        break;
      case 'request-models':
        this._view?.webview.postMessage({ type: 'model-changed', model: this._state.selectedModel ?? null });
        break;
      case 'open-settings':
        vscode.commands.executeCommand('omlx.openSettingsPanel');
        break;
      case 'insert-code': {
        const editor = vscode.window.activeTextEditor;
        if (editor) {
          editor.edit((eb) => eb.insert(editor.selection.active, msg.code));
        }
        break;
      }
      case 'clear-history':
        break;
    }
  }

  private post(msg: unknown): void {
    this._view?.webview.postMessage(msg);
  }

  private async _runAgentLoop(userHistory: ChatMessage[]): Promise<void> {
    const model = this._state.selectedModel;
    if (!model) {
      this.post({ type: 'stream-error', message: 'No model selected. Click the status bar to pick one.' });
      return;
    }

    this._abortController?.abort();
    this._abortController = new AbortController();
    const signal = this._abortController.signal;

    const config = vscode.workspace.getConfiguration('omlx');
    const systemPrompt = config.get<string>(
      'chat.defaultSystemPrompt',
      'You are a helpful programming assistant. You have tools to execute code, run commands, and read/write files on the user\'s machine.',
    );

    // Full message history maintained server-side across tool call iterations
    const messages: ChatMessage[] = [
      { role: 'system', content: systemPrompt },
      ...userHistory,
    ];

    let lastTps: number | undefined;

    try {
      for (let iteration = 0; iteration < MAX_AGENT_ITERATIONS; iteration++) {
        if (signal.aborted) break;

        // ── Stream one turn ──────────────────────────────
        let assistantContent = '';
        const pendingToolCalls: Map<number, ToolCall> = new Map();
        let finishReason: string | null = null;

        const gen = this._client.chatCompletionsStream(messages, model, {
          signal,
          tools: TOOL_DEFINITIONS,
        });

        for await (const chunk of gen) {
          if (signal.aborted) break;

          const choice = chunk.choices[0];
          if (!choice) continue;

          const delta = choice.delta;
          finishReason = choice.finish_reason ?? finishReason;

          // Content tokens
          if (delta.content || delta.reasoning_content) {
            assistantContent += delta.content ?? '';
            this.post({
              type: 'stream-chunk',
              content: delta.content ?? '',
              reasoning: delta.reasoning_content ?? '',
            });
          }

          // Accumulate tool call deltas by index
          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              const existing = pendingToolCalls.get(tc.index);
              if (existing) {
                existing.function.arguments += tc.function?.arguments ?? '';
              } else {
                pendingToolCalls.set(tc.index, {
                  index: tc.index,
                  id: tc.id ?? `call_${tc.index}`,
                  type: 'function',
                  function: { name: tc.function?.name ?? '', arguments: tc.function?.arguments ?? '' },
                });
              }
            }
          }

          if (chunk.usage?.generation_tokens_per_second) {
            lastTps = chunk.usage.generation_tokens_per_second;
            this._state.setLastTps(lastTps);
          }
        }

        if (signal.aborted) break;

        const toolCalls = [...pendingToolCalls.values()].sort((a, b) => a.index - b.index);

        // ── No tool calls → final answer ─────────────────
        if (toolCalls.length === 0 || finishReason === 'stop') {
          this.post({ type: 'stream-done', usage: { generation_tokens_per_second: lastTps } });
          break;
        }

        // ── Tool calls → execute and continue loop ────────
        // Add assistant turn with tool_calls to history
        messages.push({ role: 'assistant', content: assistantContent || null, tool_calls: toolCalls });

        for (const tc of toolCalls) {
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(tc.function.arguments) as Record<string, unknown>;
          } catch {
            args = {};
          }

          // Tell the webview a tool is running
          this.post({ type: 'tool-start', id: tc.id, name: tc.function.name, args });

          const result = await executeTool(tc.function.name, args);

          // Send result to webview
          this.post({ type: 'tool-result', id: tc.id, output: result.output, error: result.error });

          // Add tool result to message history
          messages.push({
            role: 'tool',
            content: result.error
              ? `Error: ${result.error}\n${result.output}`
              : result.output,
            tool_call_id: tc.id,
          });
        }

        // Signal webview that tool execution is done, model is responding again
        this.post({ type: 'tool-loop-continue' });
      }
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') return;
      this.post({ type: 'stream-error', message: e instanceof Error ? e.message : String(e) });
    }
  }

  private _getHtml(webview: vscode.Webview): string {
    const nonce = getNonce();
    const base = vscode.Uri.joinPath(this._extensionUri, 'src', 'webviews', 'chat');
    const cssUri = webview.asWebviewUri(vscode.Uri.joinPath(base, 'chat.css'));
    const jsUri = webview.asWebviewUri(vscode.Uri.joinPath(base, 'chat.js'));

    return getHtmlTemplate()
      .replace(/\{\{nonce\}\}/g, nonce)
      .replace(/\{\{cspSource\}\}/g, webview.cspSource)
      .replace('{{chatCssUri}}', cssUri.toString())
      .replace('{{chatJsUri}}', jsUri.toString());
  }

  dispose(): void {
    this._stateDisposable?.dispose();
    this._abortController?.abort();
  }
}

function getNonce(): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < 32; i++) result += chars.charAt(Math.floor(Math.random() * chars.length));
  return result;
}

function getHtmlTemplate(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src {{cspSource}} 'unsafe-inline'; script-src 'nonce-{{nonce}}'; font-src {{cspSource}}; img-src {{cspSource}} data:;" />
  <link rel="stylesheet" href="{{chatCssUri}}" />
  <title>oMLX Chat</title>
</head>
<body>
  <div class="header">
    <div class="model-badge" id="modelBadge">No model selected</div>
    <div class="header-actions">
      <button class="icon-btn" id="clearBtn" title="Clear chat history">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>
      </button>
      <button class="icon-btn" id="settingsBtn" title="Open settings">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      </button>
    </div>
  </div>

  <div class="messages" id="messages">
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      </div>
      <p>Ask oMLX anything</p>
      <p class="empty-hint">Select a model to get started</p>
    </div>
  </div>

  <div class="input-area">
    <div class="error-banner hidden" id="errorBanner"></div>
    <div class="input-row">
      <textarea id="inputBox" placeholder="Ask anything\u2026 (Enter to send, Shift+Enter for newline)" rows="1"></textarea>
      <button class="send-btn" id="sendBtn" title="Send (Enter)">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
      </button>
      <button class="cancel-btn hidden" id="cancelBtn" title="Cancel">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
      </button>
    </div>
    <div class="footer-info">
      <span id="tpsInfo" class="tps-info hidden"></span>
    </div>
  </div>

  <script nonce="{{nonce}}" src="{{chatJsUri}}"></script>
</body>
</html>`;
}
