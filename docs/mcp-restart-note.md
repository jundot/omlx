# MCP Restart Note

After installing optional MCP support, restart the managed oMLX server so the runtime can discover the newly installed dependency.

Homebrew-managed install:

```bash
/opt/homebrew/opt/omlx/libexec/bin/pip install mcp
omlx restart
```

Source install:

```bash
make mcp
omlx restart
```

Restarting avoids a confusing state where the package is present on disk but the already-running server process has not loaded MCP support yet.
