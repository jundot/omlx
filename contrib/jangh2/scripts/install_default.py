#!/usr/bin/env python3
"""Make an EXISTING, working JANGH2 environment the user's default omlx command.

No pip installation, source edits, model copying, credential access, automatic
service start, or system-wide changes. Stop the current server before installation.
The optional LaunchAgent is independent of Homebrew/the macOS application.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import plistlib
import shlex
import shutil
import subprocess
import tempfile
import uuid

LABEL = 'local.omlx.jangh2'


def render_wrapper(*, home, checkout, bridge, model_root, target_model,
                   base_path, port, profile=None, qualification=None):
    """Render paths as shell literals; model/config names are never shell code."""
    literals = {
        'CHECKOUT': checkout, 'BRIDGE': bridge, 'MODEL_ROOT': model_root,
        'TARGET_MODEL': target_model, 'BASE_PATH': base_path, 'PORT': str(port),
        'AGENT': Path(home)/'Library/LaunchAgents'/f'{LABEL}.plist',
        'LABEL': LABEL, 'PROFILE': profile or '', 'QUALIFICATION': qualification or '',
    }
    text = '#!/bin/bash\nset -euo pipefail\n'
    for key,value in literals.items():
        text += key+'='+shlex.quote(str(value))+'\n'
    text += r'''
PY="$CHECKOUT/.venv/bin/python"
DOMAIN="gui/$(id -u)"
SERVICE="$DOMAIN/$LABEL"
LOG="$BASE_PATH/logs/jangh2-service.log"

start_service() {
    if ! command -v launchctl >/dev/null 2>&1; then
        echo "Managed service commands require macOS launchctl." >&2
        return 1
    fi
    if [ ! -f "$AGENT" ]; then
        echo "No JANGH2 LaunchAgent installed. Re-run the installer with --with-service." >&2
        return 1
    fi
    if launchctl print "$SERVICE" >/dev/null 2>&1; then
        echo "JANGH2 LaunchAgent is already loaded. Use 'omlx status' or 'omlx restart'."
        return 0
    fi
    if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "Port $PORT is occupied. Stop the existing server before starting JANGH2." >&2
        lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >&2 || true
        return 1
    fi
    launchctl enable "$SERVICE"
    launchctl bootstrap "$DOMAIN" "$AGENT"
    echo "LaunchAgent submitted. Check 'omlx logs' for startup success."
}

stop_service() {
    if launchctl print "$SERVICE" >/dev/null 2>&1; then
        launchctl bootout "$SERVICE"
        echo "Stopped JANGH2 LaunchAgent."
    else
        echo "JANGH2 LaunchAgent is not loaded. Stop foreground servers with Ctrl-C."
    fi
}

[ "$#" -gt 0 ] || set -- serve
case "$1" in
    start) start_service; exit $? ;;
    stop) stop_service; exit $? ;;
    restart) stop_service; start_service; exit $? ;;
    status) exec launchctl print "$SERVICE" ;;
    logs) mkdir -p "$(dirname "$LOG")"; touch "$LOG"; exec tail -n 100 -f "$LOG" ;;
    serve) shift ;;
    *)
        # Keep non-serving commands on the selected Python, not on the PATH's brew binary.
        export OMLX_BASE_PATH="$BASE_PATH"
        exec "$PY" -c 'from omlx.cli import main; main()' "$@"
        ;;
esac

# Help must not initialise GPU hooks just to display CLI options.
for arg in "$@"; do
    case "$arg" in
        -h|--help) exec "$PY" -c 'from omlx.cli import main; main()' serve "$@" ;;
    esac
done

for file in "$PY" "$BRIDGE/serve.py" "$TARGET_MODEL/config.json"; do
    if [ ! -f "$file" ]; then
        echo "Missing required file: $file" >&2
        exit 1
    fi
done
if [ ! -d "$MODEL_ROOT" ]; then
    echo "Missing model discovery root: $MODEL_ROOT" >&2
    exit 1
fi

# These defaults are embedded so background launchd does not depend on .zshrc.
# Explicit environment values retain priority for foreground experiments.
export OMLX_JANGH_BACKEND="${OMLX_JANGH_BACKEND:-metal}"
export OMLX_JANGH_THINKING="${OMLX_JANGH_THINKING:-off}"
export OMLX_JANGH_PLE_CACHE_MB="${OMLX_JANGH_PLE_CACHE_MB:-16}"
export OMLX_JANGH_PLE_DEVICE_MB="${OMLX_JANGH_PLE_DEVICE_MB:-0}"
export OMLX_JANGH_PROFILE="${OMLX_JANGH_PROFILE-$PROFILE}"
export OMLX_JANGH_QUALIFICATION="${OMLX_JANGH_QUALIFICATION-$QUALIFICATION}"
export OMLX_BASE_PATH="$BASE_PATH"

exec "$PY" "$BRIDGE/serve.py" \
    --checkout "$CHECKOUT" \
    --model "$TARGET_MODEL" \
    --backend "$OMLX_JANGH_BACKEND" \
    serve --base-path "$BASE_PATH" \
    --model-dir "$MODEL_ROOT" --host 127.0.0.1 --port "$PORT" "$@"
'''
    return text


def launch_agent(*, home, wrapper, base_path):
    return {
        'Label': LABEL,
        'ProgramArguments': [str(wrapper),'serve'],
        'WorkingDirectory': str(home),
        'RunAtLoad': True,
        'KeepAlive': True,
        'ThrottleInterval': 30,
        'ExitTimeOut': 60,
        'StandardOutPath': str(Path(base_path)/'logs/jangh2-service.log'),
        'StandardErrorPath': str(Path(base_path)/'logs/jangh2-service.log'),
        'EnvironmentVariables': {
            'HOME': str(home),
            'PATH': '/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
            'PYTHONUNBUFFERED': '1',
        },
    }


def write_with_backup(path, content, mode):
    """Back up an existing path, then atomically replace it (never follow symlinks)."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    backup=None
    if path.exists() or path.is_symlink():
        suffix=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        backup=path.with_name(path.name+'.backup-'+suffix)
        shutil.copy2(path,backup,follow_symlinks=False)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(content);f.flush();os.fsync(f.fileno())
        os.chmod(tmp,mode)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return backup


def path_argument(value):
    return Path(value).expanduser().resolve()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout',type=path_argument,required=True)
    parser.add_argument('--bridge',type=path_argument,required=True)
    parser.add_argument('--model-root',type=path_argument,required=True)
    parser.add_argument('--target-model',type=path_argument,required=True)
    parser.add_argument('--base-path',type=path_argument,required=True)
    parser.add_argument('--port',type=int,default=8500)
    parser.add_argument('--profile',type=path_argument)
    parser.add_argument('--qualification',type=path_argument)
    parser.add_argument('--with-service',action='store_true',
                        help='Install login LaunchAgent, but do not start it now')
    args=parser.parse_args(argv)
    if not 1 <= args.port <= 65535:parser.error('port must be 1..65535')
    home=Path.home().resolve()
    python=args.checkout/'.venv/bin/python'
    required=[python,args.bridge/'serve.py',args.target_model/'config.json']
    for path in required+[p for p in (args.profile,args.qualification) if p is not None]:
        if not path.is_file():parser.error(f'Required file not found: {path}')
    if not args.model_root.is_dir():parser.error(f'No model root: {args.model_root}')
    if not os.access(python,os.X_OK):parser.error(f'Python is not executable: {python}')
    if args.with_service and os.uname().sysname!='Darwin':
        parser.error('--with-service requires macOS')
    # Verify exact package provenance without reinstalling or importing models.
    try:
        output=subprocess.check_output([str(python),'-c',
            'import pathlib,omlx; print(pathlib.Path(omlx.__file__).resolve())'],
            text=True,timeout=30)
        imported=Path(output.strip().splitlines()[-1])
    except (OSError,subprocess.SubprocessError,IndexError) as exc:
        parser.error(f'Cannot verify selected oMLX environment: {exc}')
    if not imported.is_relative_to(args.checkout/'omlx'):
        parser.error(f'oMLX imports from {imported}, not {args.checkout}/omlx')

    kwargs={k:getattr(args,k) for k in ('checkout','bridge','model_root','target_model',
             'base_path','port','profile','qualification')}
    text=render_wrapper(home=home,**kwargs)
    # Validate before changing the existing command.
    try:
        subprocess.run(['/bin/bash','-n'],input=text,text=True,check=True)
    except (OSError,subprocess.SubprocessError) as exc:
        parser.error(f'Generated launcher did not validate: {exc}')
    dest=home/'.local/bin/omlx'
    backup=write_with_backup(dest,text.encode(),0o755)
    print('Default launcher:',dest)
    print('Existing launcher backup:',backup or '(none)')
    print('Runtime retained:',imported)
    print('Model discovery root:',args.model_root)
    print('JANGH2 hook target only:',args.target_model)
    if args.with_service:
        (args.base_path/'logs').mkdir(parents=True,exist_ok=True)
        agent=home/'Library/LaunchAgents'/f'{LABEL}.plist'
        contents=plistlib.dumps(launch_agent(home=home,wrapper=dest,base_path=args.base_path))
        old=write_with_backup(agent,contents,0o644)
        print('LaunchAgent installed:',agent)
        if old:print('Previous LaunchAgent backup:',old)
        print('It will start at login. Run "omlx start" to start now after stopping other servers.')
    print('Prepend $HOME/.local/bin to PATH, then run "omlx serve".')
    print('No dependencies, model files or existing bridge/runtime sources were changed.')

if __name__=='__main__':
    main()
