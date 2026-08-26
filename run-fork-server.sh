#!/bin/sh
# Run the oMLX server from THIS fork checkout (cluster fixes) using the
# installed app's bundled Python + MLX. Same ~/.omlx config, models, and
# port (8000) as the normal app.
#
# IMPORTANT: quit the menu-bar oMLX app first (it owns port 8000 and ~/.omlx).
# When done testing, stop this (Ctrl-C) and relaunch the app.
#
# Works on any Mac where the fork is checked out at ~/repos/omlx and oMLX.app
# is installed. Run the SAME script on both cluster nodes.

set -e
# The repo this script lives in — works from a normal checkout or a worktree.
FORK="$(cd "$(dirname "$0")" && pwd)"
APP="/Applications/oMLX.app/Contents/Resources"
PYHOME="$APP/Python/cpython-3.11"
SITE="$APP/Python/framework-mlx-base/lib/python3.11/site-packages"

if [ ! -d "$FORK/omlx" ]; then
    echo "fork checkout not found at $FORK" >&2; exit 1
fi
if [ ! -x "$PYHOME/bin/python3" ]; then
    echo "bundled Python not found — is oMLX.app installed?" >&2; exit 1
fi

# Refuse to start if the app's server still holds port 8000.
if /usr/bin/nc -z -G 2 127.0.0.1 8000 >/dev/null 2>&1; then
    echo "Port 8000 is in use — quit the oMLX menu-bar app first." >&2; exit 1
fi

export PYTHONHOME="$PYHOME"
export PYTHONPATH="$FORK:$SITE"
export PYTHONDONTWRITEBYTECODE=1

echo "Serving oMLX from fork: $FORK  (branch: $(git -C "$FORK" branch --show-current 2>/dev/null))"
cd "$FORK"
exec "$PYHOME/bin/python3" -c "import sys; sys.argv=['omlx','serve']; from omlx.cli import main; sys.exit(main() or 0)"
