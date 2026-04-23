// SPDX-License-Identifier: Apache-2.0
import * as http from 'node:http';
import * as https from 'node:https';
import type { IncomingMessage } from 'node:http';
import type {
  AdminModel,
  ChatChunk,
  ChatMessage,
  GlobalSettings,
  OmlxModel,
  StatsSnapshot,
  ToolDefinition,
} from './types.js';

export interface StreamOptions {
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
  tools?: ToolDefinition[];
}

export class OmlxClient {
  constructor(
    private readonly _serverUrl: () => string,
    private readonly _apiKey: () => string,
  ) {}

  private _buildHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    };
    const key = this._apiKey();
    if (key) {
      headers['Authorization'] = `Bearer ${key}`;
    }
    return headers;
  }

  private async _get<T>(path: string): Promise<T> {
    const url = new URL(path, this._serverUrl());
    const mod = url.protocol === 'https:' ? https : http;
    return new Promise((resolve, reject) => {
      const req = mod.request(
        url,
        { method: 'GET', headers: this._buildHeaders() },
        (res) => {
          let data = '';
          res.on('data', (chunk: Buffer) => (data += chunk.toString('utf8')));
          res.on('end', () => {
            if (res.statusCode && res.statusCode >= 400) {
              reject(new Error(`HTTP ${res.statusCode} from ${path}: ${data.slice(0, 200)}`));
              return;
            }
            try {
              resolve(JSON.parse(data) as T);
            } catch {
              reject(new Error(`oMLX: invalid JSON from ${path}: ${data.slice(0, 200)}`));
            }
          });
          res.on('error', reject);
        },
      );
      req.on('error', reject);
      req.end();
    });
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const url = new URL(path, this._serverUrl());
    const mod = url.protocol === 'https:' ? https : http;
    const payload = JSON.stringify(body);
    return new Promise((resolve, reject) => {
      const req = mod.request(
        url,
        {
          method: 'POST',
          headers: { ...this._buildHeaders(), 'Content-Length': Buffer.byteLength(payload) },
        },
        (res) => {
          let data = '';
          res.on('data', (chunk: Buffer) => (data += chunk.toString('utf8')));
          res.on('end', () => {
            if (res.statusCode && res.statusCode >= 400) {
              reject(new Error(`oMLX: HTTP ${res.statusCode} from ${path}`));
              return;
            }
            try {
              resolve(data ? (JSON.parse(data) as T) : ({} as T));
            } catch {
              resolve({} as T);
            }
          });
          res.on('error', reject);
        },
      );
      req.on('error', reject);
      req.write(payload);
      req.end();
    });
  }

  async healthCheck(): Promise<boolean> {
    try {
      await this._get('/health');
      return true;
    } catch {
      return false;
    }
  }

  async listModels(): Promise<OmlxModel[]> {
    const result = await this._get<{ data: OmlxModel[] }>('/v1/models');
    return result.data ?? [];
  }

  async listAdminModels(): Promise<AdminModel[]> {
    try {
      // /admin/api/models returns a plain array
      const result = await this._get<AdminModel[] | { models: AdminModel[] }>(
        '/admin/api/models',
      );
      return Array.isArray(result) ? result : (result.models ?? []);
    } catch {
      // Fall back to OpenAI endpoint if admin auth fails
      const models = await this.listModels();
      return models.map((m) => ({
        id: m.id,
        loaded: true,
        is_loading: false,
        pinned: false,
        is_default: false,
      }));
    }
  }

  async getGlobalSettings(): Promise<GlobalSettings> {
    return this._get<GlobalSettings>('/admin/api/global-settings');
  }

  async getStats(): Promise<StatsSnapshot> {
    return this._get<StatsSnapshot>('/admin/api/stats');
  }

  async loadModel(modelId: string): Promise<void> {
    await this._post(`/admin/api/models/${encodeURIComponent(modelId)}/load`, {});
  }

  async unloadModel(modelId: string): Promise<void> {
    await this._post(`/admin/api/models/${encodeURIComponent(modelId)}/unload`, {});
  }

  async completion(prompt: string, model: string, maxTokens: number): Promise<string> {
    const result = await this._post<{
      choices: Array<{ text: string }>;
    }>('/v1/completions', {
      model,
      prompt,
      max_tokens: maxTokens,
      stream: false,
      stop: ['\n\n', '```'],
    });
    return result.choices?.[0]?.text ?? '';
  }

  async *chatCompletionsStream(
    messages: ChatMessage[],
    model: string,
    opts: StreamOptions = {},
  ): AsyncGenerator<ChatChunk> {
    const url = new URL('/v1/chat/completions', this._serverUrl());
    const mod = url.protocol === 'https:' ? https : http;

    const body = JSON.stringify({
      model,
      messages,
      stream: true,
      stream_options: { include_usage: true },
      temperature: opts.temperature ?? 0.7,
      max_tokens: opts.maxTokens ?? 2048,
      ...(opts.tools && opts.tools.length > 0 ? { tools: opts.tools, tool_choice: 'auto' } : {}),
    });

    const headers = {
      ...this._buildHeaders(),
      Accept: 'text/event-stream',
      'Content-Length': Buffer.byteLength(body),
    };

    const response = await new Promise<IncomingMessage>((resolve, reject) => {
      const req = mod.request(url, { method: 'POST', headers }, resolve);
      req.on('error', reject);
      if (opts.signal) {
        opts.signal.addEventListener('abort', () => req.destroy());
      }
      req.write(body);
      req.end();
    });

    if (response.statusCode && response.statusCode >= 400) {
      throw new Error(`oMLX: HTTP ${response.statusCode} from /v1/chat/completions`);
    }

    let buffer = '';
    for await (const raw of response) {
      buffer += (raw as Buffer).toString('utf8');
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') {
          continue;
        }
        if (trimmed.startsWith('data: ')) {
          try {
            yield JSON.parse(trimmed.slice(6)) as ChatChunk;
          } catch {
            // Malformed chunk — skip
          }
        }
      }
    }
  }
}
