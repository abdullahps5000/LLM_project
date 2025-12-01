# Fix Android Agent Connection Issue

## Problem
- Android agent is not running
- SSH connection to Android times out
- Coordinator can't discover Android agent

## Solution

### Step 1: Start SSH on Android Phone

On your Android phone in Termux:

```bash
# 1. Start SSH daemon
sshd

# 2. Check if it's running
ps aux | grep sshd

# 3. Check what port SSH is using (usually 8022)
netstat -tuln | grep ssh

# 4. Make sure you set a password (if not already done)
passwd
```

### Step 2: Verify SSH Connection from PC

On your PC:

```bash
# Test SSH connection
ssh -p 8022 u0_a123@172.20.10.3 "echo 'Connected successfully'"

# If this fails, try:
# - Check if the username is correct (run 'whoami' in Termux)
# - Check if SSH port is 8022 (run 'netstat -tuln | grep ssh' in Termux)
# - Make sure Android and PC are on the same WiFi network
```

### Step 3: Start Android Agent Manually (if SSH works)

If SSH works, you can start the agent manually:

**Option A: Via SSH (from PC)**
```bash
ssh -p 8022 u0_a123@172.20.10.3 "cd ~/LLM_project && source .venv/bin/activate && ./start_android_agent.sh"
```

**Option B: Directly on Android (in Termux)**
```bash
cd ~/LLM_project
source .venv/bin/activate
./start_android_agent.sh
```

### Step 4: Verify Agent is Running

On Android (in Termux):
```bash
# Check if agent is running
ps aux | grep agent_main

# Check logs
cat ~/LLM_project/.android_agent.log

# Test the agent endpoint
curl http://127.0.0.1:8008/v1/health
```

On PC:
```bash
# Test from PC
curl http://172.20.10.3:8008/v1/health

# Check status
./manage.sh status
```

### Step 5: If SSH Port is Different

If your Android SSH is on a different port (not 8022), update `manage.sh`:

```bash
# Edit manage.sh, find this line:
ANDROID_PORT="${ANDROID_PORT:-8022}"

# Change 8022 to your actual SSH port
```

### Step 6: Run Coordinator Again

Once Android agent is running:

```bash
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008,http://172.20.10.3:8008" \
  --pipeline-order "pc,pi,android" \
  --min-prefix 4 \
  --mem-fraction 0.40 \
  --package
```

## Alternative: Run Without Android

If you can't get Android working, you can run with just PC + Pi:

```bash
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --min-prefix 4 \
  --mem-fraction 0.40 \
  --package
```

