#!/usr/bin/env python3
"""Move all self-sent emails from INBOX to 'readings' using himalaya."""

import json
import subprocess

EMAIL = "czha168@gmail.com"
FOLDER = "readings"

# 1. Create folder (ignore error if it already exists)
subprocess.run(["himalaya", "folder", "create", FOLDER], capture_output=True)

# 2. List all envelopes in INBOX
r = subprocess.run(
    ["himalaya", "envelope", "list", "-f", "INBOX", "--page-size", "100", "-o", "json"],
    capture_output=True, text=True, timeout=30,
)
envelopes = json.loads(r.stdout)

# 3. Find self-sent (from == to == EMAIL)
ids = [
    str(e["id"])
    for e in envelopes
    if e.get("from", {}).get("addr") == EMAIL and e.get("to", {}).get("addr") == EMAIL
]
print(f"Found {len(ids)} self-sent email(s).")

# 4. Move them all at once
if ids:
    subprocess.run(
        ["himalaya", "message", "move", FOLDER] + ids,
        capture_output=True, text=True, timeout=60,
    )
    print(f"Moved {len(ids)} email(s) to '{FOLDER}'.")
else:
    print("Nothing to move.")
