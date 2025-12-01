#!/bin/bash
# Start Pi Agent in background
# This script runs on the Pi

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.pi_agent.pid"
LOG_FILE="$SCRIPT_DIR/.pi_agent.log"

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    source .venv/bin/activate
fi

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "Pi Agent already running (PID: $(cat "$PID_FILE"))"
    exit 0
fi

echo "Starting Pi Agent..."
nohup python -m ebp.agent_main --name pi --port 8008 > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "✓ Pi Agent started (PID: $(cat "$PID_FILE"))"
    echo "Logs: $LOG_FILE"
else
    echo "✗ Pi Agent failed to start. Check $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

