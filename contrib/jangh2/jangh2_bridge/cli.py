"""Installed convenience entry point; existing file-based launches remain valid."""
from importlib import import_module
import sys
from pathlib import Path
COMMANDS={
    'serve':'serve','qualify':'qualify','tune':'benchmark_performance',
    'compare':'compare_model','profile':'profile_model','workload':'benchmark_workload',
    'ple-benchmark':'benchmark_ple_device','snapshot':'capture_environment',
    'prepare':'prepare','device-check':'check_device_cache',
}

def main():
    if len(sys.argv)<2 or sys.argv[1] in ('-h','--help'):
        print('Usage: jangh2-bridge <command> [options]\nCommands: '+', '.join(COMMANDS))
        return 0
    command=sys.argv[1]
    if command not in COMMANDS:raise SystemExit('Unknown command: '+command)
    old=sys.argv;old_path=list(sys.path)
    try:
        sys.path.insert(0,str(Path(__file__).parent/'_tools'))
        sys.argv=[COMMANDS[command]+'.py',*old[2:]]
        return import_module('jangh2_bridge._tools.'+COMMANDS[command]).main()
    finally:
        sys.argv=old;sys.path[:]=old_path
