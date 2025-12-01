#!/bin/bash
# Quick script to sync code to Android phone (via Termux/SSH)
# Usage: ./sync_to_android.sh [ANDROID_IP] [ANDROID_USER]

set -e

# Configuration - EDIT THESE or pass as arguments
ANDROID_IP="${1:-${ANDROID_IP:-172.20.10.3}}"
ANDROID_USER="${2:-${ANDROID_USER:-u0_a123}}"  # Default Termux user (usually u0_aXXX)
ANDROID_PATH="${ANDROID_PATH:-/data/data/com.termux/files/home/LLM_project}"
ANDROID_PORT="${ANDROID_PORT:-8022}"  # Termux SSH default port

# Check if IP is provided
if [ -z "$ANDROID_IP" ]; then
    echo "ERROR: Android IP address not provided"
    echo ""
    echo "Usage: ./sync_to_android.sh <ANDROID_IP> [USER]"
    echo "   OR: export ANDROID_IP=<ip> && ./sync_to_android.sh"
    echo ""
    echo "To find your Android IP:"
    echo "  1. In Termux: ip addr show wlan0 | grep 'inet '"
    echo "  2. Or check WiFi settings on your phone"
    exit 1
fi

echo "Syncing to Android phone at $ANDROID_USER@$ANDROID_IP:$ANDROID_PATH"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "ERROR: rsync not found. Install with: sudo apt install rsync"
    exit 1
fi

# Check if SSH is available on Android
echo "Checking SSH connection to Android..."
if ! ssh -p "$ANDROID_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no "$ANDROID_USER@$ANDROID_IP" "echo 'SSH connection successful'" 2>/dev/null; then
    echo ""
    echo "⚠️  SSH connection failed. Please set up SSH on your Android phone:"
    echo ""
    echo "In Termux on your Android phone, run:"
    echo "  1. sshd  (starts SSH server)"
    echo "  2. passwd  (set a password for SSH)"
    echo "  3. whoami  (note your username, usually u0_aXXX)"
    echo ""
    echo "Then update ANDROID_USER in this script if needed."
    echo ""
    echo "To find your IP in Termux:"
    echo "  ip addr show wlan0 | grep 'inet '"
    exit 1
fi

echo "✓ SSH connection successful"
echo ""

# Create project directory on Android
echo "Creating project directory on Android..."
ssh -p "$ANDROID_PORT" "$ANDROID_USER@$ANDROID_IP" "mkdir -p $ANDROID_PATH/ebp"

# Sync agent files and dependencies
echo "Syncing agent files and dependencies..."

AGENT_FILES=(
  "ebp/__init__.py"
  "ebp/agent_main.py"
  "ebp/agent_app.py"
  "ebp/models.py"
  "ebp/common.py"
  "ebp/logging_config.py"
  "ebp/config.py"
  "ebp/errors.py"
  "ebp/retry.py"
  "ebp/stage_model.py"
  "ebp/kv_cache.py"
)

# Sync each file
for file in "${AGENT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  Syncing $file..."
        rsync -avz -e "ssh -p $ANDROID_PORT" "$file" "$ANDROID_USER@$ANDROID_IP:$ANDROID_PATH/$file"
    fi
done

# Sync requirements
echo ""
echo "Syncing requirements-agent.txt..."
rsync -avz -e "ssh -p $ANDROID_PORT" requirements-agent.txt "$ANDROID_USER@$ANDROID_IP:$ANDROID_PATH/"

# Sync management scripts
echo ""
echo "Syncing management scripts..."
rsync -avz -e "ssh -p $ANDROID_PORT" start_android_agent.sh "$ANDROID_USER@$ANDROID_IP:$ANDROID_PATH/" 2>/dev/null || echo "  (start_android_agent.sh will be created on Android)"
rsync -avz -e "ssh -p $ANDROID_PORT" stop_android_agent.sh "$ANDROID_USER@$ANDROID_IP:$ANDROID_PATH/" 2>/dev/null || echo "  (stop_android_agent.sh will be created on Android)"

# Make scripts executable
ssh -p "$ANDROID_PORT" "$ANDROID_USER@$ANDROID_IP" "chmod +x $ANDROID_PATH/start_android_agent.sh $ANDROID_PATH/stop_android_agent.sh 2>/dev/null || true"

echo ""
echo "✅ Sync complete!"
echo ""
echo "To start Android Agent:"
echo "  ./manage.sh start-android"
echo ""
echo "Or manually from Android (in Termux):"
echo "  cd $ANDROID_PATH"
echo "  source .venv/bin/activate"
echo "  ./start_android_agent.sh"
echo ""
echo "To check status:"
echo "  ./manage.sh status"

