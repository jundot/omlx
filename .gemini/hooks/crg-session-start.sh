#!/usr/bin/env bash
# code-review-graph: session start status (Gemini CLI hook)
# Must output ONLY JSON on stdout. Logs go to stderr. Never blocks the session.
set -euo pipefail

cat > /dev/null || true

msg="$(code-review-graph status --repo "/Users/czha168/workspaces/omlx" 2>&1 | head -n 1 || true)"

CRG_MSG="$msg" python3 -c '
import json,os
m=os.environ.get("CRG_MSG","")
print(json.dumps({"systemMessage":m,"suppressOutput":True}))
' 2>/dev/null || echo '{"suppressOutput": true}'
exit 0
