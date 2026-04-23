# oMLX — VS Code Extension

A VS Code extension for [oMLX](https://github.com/jundot/omlx), a high-performance local LLM inference server for Apple Silicon.

## Features

- **Inline code completions** — Copilot-style ghost-text completions powered by any model loaded in oMLX
- **Chat sidebar** — Streaming chat panel with full conversation history
- **Agentic tool loop** — The model can execute code, run terminal commands, read/write files, start servers, and open live previews
- **Model switcher** — Quick-pick palette showing loaded/unloaded models with size info
- **Status bar** — Live connection status and tokens-per-second display
- **Settings panel** — GUI to configure the server URL, API key, and global model parameters

## Requirements

- [oMLX server](https://github.com/jundot/omlx) running locally (default: `http://localhost:8000`)
- VS Code 1.85 or later
- macOS with Apple Silicon (M1/M2/M3/M4)

## Installation

### Development (F5)

```bash
cd extensions/vscode
npm install
npm run build
```

Then press **F5** in VS Code to launch the Extension Development Host.

### VSIX package

```bash
npm install -g @vscode/vsce
cd extensions/vscode
npm run build
vsce package
```

Install the generated `.vsix` via **Extensions: Install from VSIX…** in the command palette.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `omlx.serverUrl` | `http://localhost:8000` | oMLX server base URL |
| `omlx.apiKey` | _(empty)_ | Bearer token if the server requires authentication |
| `omlx.completions.enabled` | `true` | Enable/disable inline completions |
| `omlx.completions.maxTokens` | `128` | Maximum tokens per completion |
| `omlx.completions.debounceMs` | `300` | Delay before triggering a completion request |
| `omlx.completions.contextLines` | `20` | Lines of context sent to the model |
| `omlx.chat.defaultSystemPrompt` | _(built-in)_ | System prompt prepended to every chat session |

If your server requires an API key, set `omlx.apiKey` in VS Code settings (`Cmd+,` → search `omlx.apiKey`) or open the oMLX Settings panel via the command palette.

## Commands

| Command | Description |
|---|---|
| `oMLX: Select Model` | Open the model quick-pick switcher |
| `oMLX: Open Chat` | Focus the chat sidebar |
| `oMLX: Open Settings` | Open the settings panel |
| `oMLX: Toggle Completions` | Enable or disable inline completions |
| `oMLX: Check Server Health` | Ping the server and show a status notification |

## Agent Tools

When using the chat panel the model can invoke the following tools:

| Tool | Description |
|---|---|
| `execute_code` | Run Python, JavaScript, TypeScript, or shell code in a temp file |
| `run_command` | Execute an arbitrary shell command (requires confirmation) |
| `read_file` | Read a file relative to the workspace root |
| `write_file` | Write or overwrite a file (requires confirmation) |
| `list_files` | List files matching a glob pattern |
| `open_preview` | Open a URL in the VS Code Simple Browser |
| `start_server` | Spawn a long-running process and wait for a port to be ready |

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) at the repo root. The extension source lives in `extensions/vscode/src/`.

```
src/
  api/          HTTP client and type definitions
  commands/     VS Code command registrations
  panels/       Chat and Settings webview panels
  providers/    Inline completion and status bar providers
  state/        Shared extension state singleton
  tools/        Agent tool definitions and executor
  webviews/     Webview HTML, CSS, and JS assets
  extension.ts  Activation entry point
```

## License

Apache 2.0 — see [LICENSE](../../LICENSE).
