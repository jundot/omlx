// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { OmlxClient } from '../api/client.js';
import type { ExtensionState } from '../state/ExtensionState.js';

export class OmlxInlineCompletionProvider
  implements vscode.InlineCompletionItemProvider
{
  private _debounceTimer: ReturnType<typeof setTimeout> | undefined;
  private _abortController: AbortController | undefined;

  constructor(
    private readonly _client: OmlxClient,
    private readonly _state: ExtensionState,
  ) {}

  async provideInlineCompletionItems(
    document: vscode.TextDocument,
    position: vscode.Position,
    _context: vscode.InlineCompletionContext,
    token: vscode.CancellationToken,
  ): Promise<vscode.InlineCompletionList | undefined> {
    const config = vscode.workspace.getConfiguration('omlx');
    if (!config.get<boolean>('completions.enabled', true)) {
      return undefined;
    }

    const model = this._state.selectedModel;
    if (!model || !this._state.serverHealthy) {
      return undefined;
    }

    // Cancel any previous in-flight request
    this._abortController?.abort();

    // Debounce
    await new Promise<void>((resolve, reject) => {
      if (this._debounceTimer) {
        clearTimeout(this._debounceTimer);
      }
      const debounceMs = config.get<number>('completions.debounceMs', 300);
      this._debounceTimer = setTimeout(resolve, debounceMs);
      token.onCancellationRequested(() => {
        clearTimeout(this._debounceTimer);
        reject(new Error('cancelled'));
      });
    }).catch(() => undefined);

    if (token.isCancellationRequested) {
      return undefined;
    }

    const contextLines = config.get<number>('completions.contextLines', 100);
    const maxTokens = config.get<number>('completions.maxTokens', 80);

    const startLine = Math.max(0, position.line - contextLines);
    const prefix = document.getText(
      new vscode.Range(
        new vscode.Position(startLine, 0),
        position,
      ),
    );

    if (!prefix.trim()) {
      return undefined;
    }

    this._abortController = new AbortController();
    try {
      const completion = await this._client.completion(prefix, model, maxTokens);
      if (!completion || token.isCancellationRequested) {
        return undefined;
      }

      return new vscode.InlineCompletionList([
        new vscode.InlineCompletionItem(
          completion,
          new vscode.Range(position, position),
        ),
      ]);
    } catch {
      return undefined;
    }
  }

  dispose(): void {
    if (this._debounceTimer) {
      clearTimeout(this._debounceTimer);
    }
    this._abortController?.abort();
  }
}
