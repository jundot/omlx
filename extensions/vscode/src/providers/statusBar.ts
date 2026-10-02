// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';
import type { ExtensionState } from '../state/ExtensionState.js';

export class StatusBarManager {
  private readonly _modelItem: vscode.StatusBarItem;
  private readonly _tpsItem: vscode.StatusBarItem;
  private _tpsHideTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(private readonly _state: ExtensionState) {
    this._modelItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Left,
      100,
    );
    this._modelItem.command = 'omlx.selectModel';
    this._modelItem.show();

    this._tpsItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100,
    );

    this._update();
  }

  private _update(): void {
    const { serverHealthy, selectedModel, lastTps } = this._state;

    if (!serverHealthy) {
      this._modelItem.text = '$(circle-slash) oMLX: offline';
      this._modelItem.tooltip = 'oMLX server is not reachable. Click to select model.';
      this._modelItem.color = new vscode.ThemeColor('statusBarItem.warningForeground');
      this._modelItem.backgroundColor = new vscode.ThemeColor(
        'statusBarItem.warningBackground',
      );
    } else if (!selectedModel) {
      this._modelItem.text = '$(circle-outline) oMLX: no model';
      this._modelItem.tooltip = 'Click to select a model';
      this._modelItem.color = undefined;
      this._modelItem.backgroundColor = undefined;
    } else {
      this._modelItem.text = `$(circle-filled) ${selectedModel}`;
      this._modelItem.tooltip = `oMLX: ${selectedModel} — click to switch model`;
      this._modelItem.color = undefined;
      this._modelItem.backgroundColor = undefined;
    }

    if (lastTps !== undefined) {
      this._tpsItem.text = `$(zap) ${lastTps.toFixed(1)} tok/s`;
      this._tpsItem.show();

      if (this._tpsHideTimer) {
        clearTimeout(this._tpsHideTimer);
      }
      this._tpsHideTimer = setTimeout(() => {
        this._tpsItem.hide();
        this._tpsHideTimer = undefined;
      }, 10_000);
    }
  }

  notifyStateChanged(): void {
    this._update();
  }

  dispose(): void {
    if (this._tpsHideTimer) {
      clearTimeout(this._tpsHideTimer);
    }
    this._modelItem.dispose();
    this._tpsItem.dispose();
  }
}
