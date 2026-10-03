"""Compare explicit EXL3 server configurations using synthetic API requests.

Run once with MTP off, then restart/reload with MTP on and run with --compare.
This script does not load weights itself or change settings/services. Keep the
server's normal memory guards enabled and inspect its MTP acceptance logs.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "coding": "Write a Python function that merges overlapping inclusive intervals. Include one example. Be concise.",
    "chat": "Our mate built a boxing game but the character has no arms. Give a funny short reaction.",
    "reasoning": "A bat and ball cost £1.10 together. The bat costs £1 more than the ball. Explain the ball price briefly.",
    "long": "Write a concise Python LRU cache using OrderedDict with get and put methods. Explain eviction.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    key = os.environ.get("OMLX_API_KEY", "")

    def chat(text, **extra):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0,
            "max_tokens": 128,
            "stream": False,
            **extra,
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        # Random call IDs are transport metadata, not output-parity evidence.
        for call in message.get("tool_calls", []):
            call.pop("id", None)
        return {
            "seconds": time.monotonic() - start,
            "usage": result.get("usage"),
            "finish_reason": result["choices"][0].get("finish_reason"),
            "message": message,
        }

    chat("What is 2+2? Only the number.")  # Exclude load/first-use compilation.
    results = {name: chat(text) for name, text in PROMPTS.items()}
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    results["tool"] = chat(
        "Use get_weather for London.",
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    args.output.write_text(json.dumps({"label": args.label, "results": results}, indent=2))
    previous = json.loads(args.compare.read_text())["results"] if args.compare else {}
    for name, result in results.items():
        summary = {"request": name, "seconds": round(result["seconds"], 3)}
        if name in previous:
            summary["wall_speedup"] = round(previous[name]["seconds"] / result["seconds"], 3)
            summary["message_equal"] = previous[name]["message"] == result["message"]
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
