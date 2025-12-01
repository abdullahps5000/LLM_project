#!/bin/bash
# Quick script to sync code to Raspberry Pi

set -e

# Configuration - EDIT THESE
PI_IP="${PI_IP:-172.20.10.2}"  # Your Pi's IP address
PI_USER="${PI_USER:-abdoulaye}"          # Your Pi username
PI_PATH="${PI_PATH:-/home/$PI_USER/LLM_project}"  # Path on Pi
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI_PASS="${PI_PASS:-234879}"  # Pi password (optional, can use SSH keys instead)

echo "Syncing to Raspberry Pi at $PI_USER@$PI_IP:$PI_PATH"
echo ""

# Check if rsync is available
if ! command -v rsync &> /dev/null; then
    echo "ERROR: rsync not found. Install with: sudo apt install rsync"
    exit 1
fi

# Use sshpass for password authentication
echo "Using sshpass for password authentication"
echo ""

# Sync agent files and dependencies
echo "Syncing agent files and dependencies..."
echo "  - agent_main.py, agent_app.py, models.py, common.py"
echo "  - logging_config.py, config.py, errors.py, retry.py"
echo "  - stage_model.py, session_manager.py, serialization.py, validation.py"

# Create temp list of files to sync
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
  "ebp/session_manager.py"
  "ebp/serialization.py"
  "ebp/validation.py"
)

# Sync each file
for file in "${AGENT_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        sshpass -p "$PI_PASS" rsync -avz "$PROJECT_ROOT/$file" $PI_USER@$PI_IP:$PI_PATH/"$file"
    fi
done

# Ensure ebp directory exists on Pi
sshpass -p "$PI_PASS" ssh $PI_USER@$PI_IP "mkdir -p $PI_PATH/ebp"

# Sync requirements
echo ""
echo "Syncing requirements-agent.txt..."
sshpass -p "$PI_PASS" rsync -avz "$PROJECT_ROOT/requirements-agent.txt" $PI_USER@$PI_IP:$PI_PATH/

# Sync management scripts
echo ""
echo "Syncing management scripts..."
sshpass -p "$PI_PASS" rsync -avz "$PROJECT_ROOT/scripts/start_pi_agent.sh" $PI_USER@$PI_IP:$PI_PATH/
sshpass -p "$PI_PASS" rsync -avz "$PROJECT_ROOT/scripts/stop_pi_agent.sh" $PI_USER@$PI_IP:$PI_PATH/
sshpass -p "$PI_PASS" ssh $PI_USER@$PI_IP "chmod +x $PI_PATH/start_pi_agent.sh $PI_PATH/stop_pi_agent.sh"

echo ""
echo "✅ Sync complete!"
echo ""
echo "To start Pi Agent:"
echo "  ssh $PI_USER@$PI_IP 'cd $PI_PATH && ./start_pi_agent.sh'"
echo ""
echo "Or use the unified manager:"
echo "  ./manage.sh start  (starts PC services)"
echo "  ./manage.sh status (check all services)"

