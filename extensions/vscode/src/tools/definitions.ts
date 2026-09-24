// SPDX-License-Identifier: Apache-2.0
import type { ToolDefinition } from '../api/types.js';

export const TOOL_DEFINITIONS: ToolDefinition[] = [
  {
    type: 'function',
    function: {
      name: 'execute_code',
      description:
        'Execute code locally and return stdout/stderr. Use this to run calculations, test logic, process data, or produce results. Runs in a temp file with a 30-second timeout.',
      parameters: {
        type: 'object',
        properties: {
          language: {
            type: 'string',
            enum: ['python', 'javascript', 'typescript', 'bash'],
            description: 'Programming language to use',
          },
          code: {
            type: 'string',
            description: 'The code to execute',
          },
        },
        required: ['language', 'code'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'run_command',
      description:
        'Run a shell command in the workspace directory and return its output. Use for git, npm, file operations, etc.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'Shell command to run',
          },
        },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_file',
      description: 'Read the contents of a file. Path can be relative to the workspace root or absolute.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path to read',
          },
        },
        required: ['path'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'write_file',
      description: 'Write or overwrite a file. Path can be relative to workspace root or absolute.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'File path to write',
          },
          content: {
            type: 'string',
            description: 'Content to write to the file',
          },
        },
        required: ['path', 'content'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'list_files',
      description: 'List files and directories at a given path.',
      parameters: {
        type: 'object',
        properties: {
          path: {
            type: 'string',
            description: 'Directory path (relative to workspace root or absolute). Defaults to workspace root.',
          },
        },
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'open_preview',
      description:
        'Open a URL in the VS Code built-in browser panel (Simple Browser). Use after starting a server to show the result inline without opening an external browser.',
      parameters: {
        type: 'object',
        properties: {
          url: {
            type: 'string',
            description: 'The URL to preview, e.g. http://localhost:3000',
          },
          title: {
            type: 'string',
            description: 'Optional panel title shown in the tab',
          },
        },
        required: ['url'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'start_server',
      description:
        'Spawn a long-running background process (e.g. a dev server) and wait until the given port is accepting connections before returning. Use run_command for one-shot commands; use this for servers that stay running.',
      parameters: {
        type: 'object',
        properties: {
          command: {
            type: 'string',
            description: 'Shell command to start the server, e.g. "python app.py" or "npm run dev"',
          },
          port: {
            type: 'number',
            description: 'TCP port to poll until the server is ready',
          },
          timeout_seconds: {
            type: 'number',
            description: 'How long to wait for the port to open before giving up. Default 30.',
          },
        },
        required: ['command', 'port'],
      },
    },
  },
];
