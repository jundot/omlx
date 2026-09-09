// SPDX-License-Identifier: Apache-2.0
import * as child_process from 'node:child_process';
import * as fs from 'node:fs/promises';
import * as net from 'node:net';
import * as os from 'node:os';
import * as nodePath from 'node:path';
import * as vscode from 'vscode';

const MAX_OUTPUT_CHARS = 8_000;
const EXEC_TIMEOUT_MS = 30_000;

export interface ToolResult {
  output: string;
  error?: string;
}

function workspaceRoot(): string {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? os.homedir();
}

function resolvePath(p: string): string {
  if (nodePath.isAbsolute(p)) return p;
  return nodePath.join(workspaceRoot(), p);
}

function truncate(s: string): string {
  if (s.length <= MAX_OUTPUT_CHARS) return s;
  return s.slice(0, MAX_OUTPUT_CHARS) + `\n… (truncated, ${s.length} chars total)`;
}

function exec(cmd: string, cwd: string): Promise<ToolResult> {
  return new Promise((resolve) => {
    child_process.exec(
      cmd,
      { cwd, timeout: EXEC_TIMEOUT_MS, maxBuffer: 4 * 1024 * 1024 },
      (err, stdout, stderr) => {
        const out = truncate(stdout.trim());
        const err_out = truncate(stderr.trim());
        if (err && !stdout) {
          resolve({ output: err_out || err.message, error: err.message });
        } else {
          const combined = [out, err_out].filter(Boolean).join('\n--- stderr ---\n');
          resolve({ output: combined || '(no output)' });
        }
      },
    );
  });
}

async function executeCode(language: string, code: string): Promise<ToolResult> {
  const ext: Record<string, string> = {
    python: '.py',
    javascript: '.js',
    typescript: '.ts',
    bash: '.sh',
  };
  const runner: Record<string, string> = {
    python: 'python3',
    javascript: 'node',
    typescript: 'npx ts-node --skipProject',
    bash: 'bash',
  };

  const fileExt = ext[language] ?? '.txt';
  const cmd = runner[language];
  if (!cmd) {
    return { output: '', error: `Unsupported language: ${language}` };
  }

  const tmpFile = nodePath.join(os.tmpdir(), `omlx_exec_${Date.now()}${fileExt}`);
  try {
    await fs.writeFile(tmpFile, code, 'utf8');
    const result = await exec(`${cmd} "${tmpFile}"`, workspaceRoot());
    return result;
  } finally {
    await fs.unlink(tmpFile).catch(() => undefined);
  }
}

async function runCommand(command: string): Promise<ToolResult> {
  const confirmed = await vscode.window.showWarningMessage(
    `oMLX wants to run: \`${command}\``,
    { modal: true },
    'Run',
  );
  if (confirmed !== 'Run') {
    return { output: '', error: 'User cancelled command execution.' };
  }
  return exec(command, workspaceRoot());
}

async function readFile(filePath: string): Promise<ToolResult> {
  const resolved = resolvePath(filePath);
  try {
    const content = await fs.readFile(resolved, 'utf8');
    return { output: truncate(content) };
  } catch (e) {
    return { output: '', error: `Cannot read ${resolved}: ${e instanceof Error ? e.message : String(e)}` };
  }
}

async function writeFile(filePath: string, content: string): Promise<ToolResult> {
  const resolved = resolvePath(filePath);
  const confirmed = await vscode.window.showWarningMessage(
    `oMLX wants to write to: ${resolved}`,
    { modal: true },
    'Write',
  );
  if (confirmed !== 'Write') {
    return { output: '', error: 'User cancelled file write.' };
  }
  try {
    await fs.mkdir(nodePath.dirname(resolved), { recursive: true });
    await fs.writeFile(resolved, content, 'utf8');
    return { output: `Written ${content.length} chars to ${resolved}` };
  } catch (e) {
    return { output: '', error: `Cannot write ${resolved}: ${e instanceof Error ? e.message : String(e)}` };
  }
}

async function listFiles(dirPath: string = '.'): Promise<ToolResult> {
  const resolved = resolvePath(dirPath);
  try {
    const entries = await fs.readdir(resolved, { withFileTypes: true });
    const lines = entries.map((e) => (e.isDirectory() ? `${e.name}/` : e.name));
    return { output: lines.join('\n') || '(empty directory)' };
  } catch (e) {
    return { output: '', error: `Cannot list ${resolved}: ${e instanceof Error ? e.message : String(e)}` };
  }
}

function waitForPort(port: number, timeoutMs: number): Promise<void> {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    function attempt() {
      const sock = net.createConnection({ port, host: '127.0.0.1' });
      sock.once('connect', () => { sock.destroy(); resolve(); });
      sock.once('error', () => {
        sock.destroy();
        if (Date.now() >= deadline) {
          reject(new Error(`Port ${port} not open after ${timeoutMs}ms`));
        } else {
          setTimeout(attempt, 500);
        }
      });
    }
    attempt();
  });
}

// Track spawned server processes so they can outlive individual tool calls
const _serverProcesses = new Map<string, child_process.ChildProcess>();

async function openPreview(url: string, title?: string): Promise<ToolResult> {
  try {
    // VS Code Simple Browser is built-in — no extension needed
    await vscode.commands.executeCommand(
      'simpleBrowser.show',
      url,
      { preserveFocus: false, viewColumn: vscode.ViewColumn.Beside },
    );
    return { output: `Opened preview: ${url}` };
  } catch {
    // Fall back to external browser if Simple Browser isn't available
    await vscode.env.openExternal(vscode.Uri.parse(url));
    return { output: `Opened ${url} in external browser` };
  }
}

async function startServer(
  command: string,
  port: number,
  timeoutSeconds = 30,
): Promise<ToolResult> {
  const confirmed = await vscode.window.showWarningMessage(
    `oMLX wants to start a server: \`${command}\``,
    { modal: true },
    'Start',
  );
  if (confirmed !== 'Start') {
    return { output: '', error: 'User cancelled server start.' };
  }

  // Kill any existing process on the same key
  const key = `${port}`;
  _serverProcesses.get(key)?.kill();

  const proc = child_process.spawn(command, {
    shell: true,
    cwd: workspaceRoot(),
    detached: false,
    stdio: 'pipe',
  });

  _serverProcesses.set(key, proc);

  const logs: string[] = [];
  proc.stdout?.on('data', (d: Buffer) => logs.push(d.toString()));
  proc.stderr?.on('data', (d: Buffer) => logs.push(d.toString()));
  proc.on('error', (e) => logs.push(`Process error: ${e.message}`));

  try {
    await waitForPort(port, timeoutSeconds * 1000);
    return {
      output: `Server started on port ${port} (pid ${proc.pid}).\nEarly output:\n${logs.slice(0, 20).join('').slice(0, 1000)}`,
    };
  } catch (e) {
    proc.kill();
    _serverProcesses.delete(key);
    return {
      output: logs.join('').slice(0, 2000),
      error: e instanceof Error ? e.message : String(e),
    };
  }
}

export async function executeTool(
  name: string,
  args: Record<string, unknown>,
): Promise<ToolResult> {
  switch (name) {
    case 'execute_code':
      return executeCode(String(args.language ?? ''), String(args.code ?? ''));
    case 'run_command':
      return runCommand(String(args.command ?? ''));
    case 'read_file':
      return readFile(String(args.path ?? '.'));
    case 'write_file':
      return writeFile(String(args.path ?? ''), String(args.content ?? ''));
    case 'list_files':
      return listFiles(args.path ? String(args.path) : '.');
    case 'open_preview':
      return openPreview(String(args.url ?? ''), args.title ? String(args.title) : undefined);
    case 'start_server':
      return startServer(
        String(args.command ?? ''),
        Number(args.port ?? 8080),
        args.timeout_seconds ? Number(args.timeout_seconds) : 30,
      );
    default:
      return { output: '', error: `Unknown tool: ${name}` };
  }
}
