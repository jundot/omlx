// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { OmlxClient } from '../api/client.js';
import type { ExtensionState } from '../state/ExtensionState.js';

export async function showModelSwitcher(
  client: OmlxClient,
  state: ExtensionState,
): Promise<void> {
  const quickPick = vscode.window.createQuickPick();
  quickPick.placeholder = 'Select an oMLX model';
  quickPick.busy = true;
  quickPick.show();

  try {
    const adminModels = await client.listAdminModels();

    // Loaded models first
    const sorted = [...adminModels].sort((a, b) => {
      if (a.loaded && !b.loaded) return -1;
      if (!a.loaded && b.loaded) return 1;
      return a.id.localeCompare(b.id);
    });

    quickPick.items = sorted.map((m) => ({
      label: m.loaded ? `$(circle-filled) ${m.id}` : `$(circle-outline) ${m.id}`,
      description: m.loaded
        ? 'loaded'
        : m.is_loading
          ? 'loading…'
          : m.estimated_size_formatted ?? 'unloaded',
      detail: m.is_default ? 'default model' : undefined,
      id: m.id,
    }));
    quickPick.busy = false;

    quickPick.onDidAccept(() => {
      const selected = quickPick.selectedItems[0] as
        | (vscode.QuickPickItem & { id: string })
        | undefined;
      if (selected) {
        state.selectedModel = selected.id;
        vscode.window.setStatusBarMessage(`oMLX: model set to ${selected.id}`, 3000);
      }
      quickPick.dispose();
    });

    quickPick.onDidHide(() => quickPick.dispose());
  } catch (e) {
    quickPick.dispose();
    vscode.window.showErrorMessage(
      `oMLX: Failed to list models — ${e instanceof Error ? e.message : String(e)}`,
    );
  }
}
