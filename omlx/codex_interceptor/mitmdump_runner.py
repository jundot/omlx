"""Small module entry point for the mitmproxy dependency bundled with oMLX."""

from mitmproxy.tools.main import mitmdump

if __name__ == "__main__":  # pragma: no cover - exercised as a child process
    mitmdump()
