// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { OmlxClient } from '../api/client.js';
import type { ExtensionState } from '../state/ExtensionState.js';
import { showModelSwitcher } from './modelSwitcher.js';

export function registerCommands(
  context: vscode.ExtensionContext,
  client: OmlxClient,
  state: ExtensionState,
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand('omlx.selectModel', () =>
      showModelSwitcher(client, state),
    ),

    vscode.commands.registerCommand('omlx.openSettings', () =>
      vscode.commands.executeCommand('omlx.openSettingsPanel'),
    ),

    vscode.commands.registerCommand('omlx.openChat', () =>
      vscode.commands.executeCommand('omlx.chatView.focus'),
    ),

    vscode.commands.registerCommand('omlx.toggleCompletions', () => {
      const config = vscode.workspace.getConfiguration('omlx');
      const current = config.get<boolean>('completions.enabled', true);
      config.update(
        'completions.enabled',
        !current,
        vscode.ConfigurationTarget.Global,
      );
      vscode.window.setStatusBarMessage(
        `oMLX completions ${!current ? 'enabled' : 'disabled'}`,
        3000,
      );
    }),

    vscode.commands.registerCommand('omlx.checkHealth', async () => {
      const healthy = await client.healthCheck();
      if (healthy) {
        vscode.window.showInformationMessage('oMLX server is reachable.');
      } else {
        vscode.window.showWarningMessage(
          `oMLX server not reachable at ${vscode.workspace.getConfiguration('omlx').get<string>('serverUrl', 'http://localhost:8000')}`,
        );
      }
    }),
  );
}
