#!/bin/bash
# Start Android Agent in background
# This script runs on the Android phone (in Termux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.android_agent.pid"
LOG_FILE="$SCRIPT_DIR/.android_agent.log"

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    else
        echo "ERROR: Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate"
        exit 1
    fi
fi

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Android Agent already running (PID: $(cat "$PID_FILE"))"
    exit 0
fi

echo "Starting Android Agent..."
nohup python -m ebp.agent_main --name android --port 8008 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "✓ Android Agent started (PID: $(cat "$PID_FILE"))"
    echo "Logs: $LOG_FILE"
    echo ""
    echo "To view logs: tail -f $LOG_FILE"
    echo "To stop: ./stop_android_agent.sh"
else
    echo "✗ Android Agent failed to start. Check $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

