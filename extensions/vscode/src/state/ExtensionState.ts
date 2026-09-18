// SPDX-License-Identifier: Apache-2.0
import * as vscode from 'vscode';

type StateChangeListener = (state: ExtensionState) => void;

export class ExtensionState {
  private static _instance: ExtensionState | undefined;

  private _selectedModel: string | undefined;
  private _serverHealthy = false;
  private _lastTps: number | undefined;
  private _listeners: StateChangeListener[] = [];
  private _globalState: vscode.Memento | undefined;

  static getInstance(): ExtensionState {
    if (!ExtensionState._instance) {
      ExtensionState._instance = new ExtensionState();
    }
    return ExtensionState._instance;
  }

  static reset(): void {
    ExtensionState._instance = undefined;
  }

  init(context: vscode.ExtensionContext): void {
    this._globalState = context.globalState;
    this._selectedModel = context.globalState.get<string>('omlx.selectedModel');
  }

  get selectedModel(): string | undefined {
    return this._selectedModel;
  }

  set selectedModel(value: string | undefined) {
    this._selectedModel = value;
    this._globalState?.update('omlx.selectedModel', value);
    this._emit();
  }

  get serverHealthy(): boolean {
    return this._serverHealthy;
  }

  set serverHealthy(value: boolean) {
    if (this._serverHealthy !== value) {
      this._serverHealthy = value;
      this._emit();
    }
  }

  get lastTps(): number | undefined {
    return this._lastTps;
  }

  setLastTps(tps: number): void {
    this._lastTps = tps;
    this._emit();
  }

  onDidChange(listener: StateChangeListener): vscode.Disposable {
    this._listeners.push(listener);
    return new vscode.Disposable(() => {
      this._listeners = this._listeners.filter((l) => l !== listener);
    });
  }

  private _emit(): void {
    for (const listener of this._listeners) {
      listener(this);
    }
  }
}

export const state = ExtensionState.getInstance();
