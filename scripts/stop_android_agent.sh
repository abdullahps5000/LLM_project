#!/bin/bash
# Stop Android Agent
# This script runs on the Android phone (in Termux)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE="$SCRIPT_DIR/.android_agent.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Android Agent is not running (no PID file found)"
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "Android Agent is not running (PID $PID not found)"
    rm -f "$PID_FILE"
    exit 0
fi

echo "Stopping Android Agent (PID: $PID)..."
kill "$PID"

# Wait a bit for graceful shutdown
sleep 2

if kill -0 "$PID" 2>/dev/null; then
    echo "Force killing Android Agent..."
    kill -9 "$PID"
fi

rm -f "$PID_FILE"
echo "✓ Android Agent stopped"

