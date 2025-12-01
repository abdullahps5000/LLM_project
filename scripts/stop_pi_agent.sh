#!/bin/bash
# Stop Pi Agent
# This script runs on the Pi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.pi_agent.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Pi Agent not running"
    exit 0
fi

PID=$(cat "$PID_FILE")
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping Pi Agent (PID: $PID)..."
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
        kill -9 "$PID" 2>/dev/null || true
    fi
fi
rm -f "$PID_FILE"
echo "✓ Pi Agent stopped"

