// SPDX-License-Identifier: Apache-2.0
export interface OmlxModel {
  id: string;
  object: 'model';
  owned_by: string;
}

export interface AdminModel {
  id: string;
  loaded: boolean;
  is_loading: boolean;
  pinned: boolean;
  is_default: boolean;
  estimated_size?: number;
  estimated_size_formatted?: string;
}

export interface GlobalSettings {
  server?: { host: string; port: number; log_level: string };
  model?: { max_model_memory: string };
  scheduler?: { max_concurrent_requests: number };
  cache?: { enabled: boolean };
  auth?: { api_key_set: boolean; api_key: string };
}

export interface StatsSnapshot {
  total_requests: number;
  total_completion_tokens: number;
  avg_generation_tps: number;
  avg_prefill_tps: number;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ToolFunction {
  name: string;
  arguments: string;
}

export interface ToolCall {
  index: number;
  id: string;
  type: 'function';
  function: ToolFunction;
}

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

export interface ChatChunk {
  id: string;
  choices: Array<{
    delta: {
      role?: string;
      content?: string;
      reasoning_content?: string;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: { name?: string; arguments?: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    generation_tokens_per_second?: number;
    prompt_tokens_per_second?: number;
  };
}

export interface VscodeConfig {
  serverUrl: string;
  apiKey: string;
  completionsEnabled: boolean;
  completionsMaxTokens: number;
  completionsDebounceMs: number;
  completionsContextLines: number;
  chatDefaultSystemPrompt: string;
}
