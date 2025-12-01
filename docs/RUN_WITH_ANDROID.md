# How to Run with Android Phone

## Quick Start (3 Devices: PC + Pi + Android)

### Step 1: Setup Android Phone (One-time setup)

On your Android phone in Termux:

```bash
# Install packages
pkg update && pkg upgrade -y
pkg install -y python git curl wget rsync openssh

# Setup project
mkdir -p ~/LLM_project
cd ~/LLM_project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies
pip install fastapi==0.115.2 uvicorn==0.32.0 httpx==0.27.2 psutil==6.1.0 numpy==2.1.3 pydantic==2.9.2 safetensors==0.4.5

# Setup SSH
sshd
passwd  # Set a password
whoami   # Note your username (usually u0_aXXX)
```

### Step 2: Sync Code to Android

On your PC:

```bash
cd ~/LLM_project
source .venv/bin/activate

# Sync code (Android IP is already set to 172.20.10.3 in sync_to_android.sh)
./sync_to_android.sh
```

**Note**: If your Termux username is different from `u0_a123`, update it:
```bash
export ANDROID_USER=<your_termux_username>
./sync_to_android.sh
```

### Step 3: Start All Agents

On your PC:

```bash
# Start PC Agent and File Server
./manage.sh start

# Start Pi Agent (via SSH)
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'

# Start Android Agent
./manage.sh start-android
```

### Step 4: Verify All Agents Are Running

```bash
./manage.sh status
```

You should see:
- ✓ PC Agent is running
- ✓ Pi Agent is running  
- ✓ Android Agent is running

### Step 5: Run Coordinator with 3 Devices

```bash
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008,http://172.20.10.3:8008" \
  --pipeline-order "pc,pi,android" \
  --min-prefix 4 \
  --mem-fraction 0.45 \
  --package
```

The coordinator will:
1. Discover all 3 agents
2. Profile their capabilities (CPU, RAM, performance)
3. Partition the model across all 3 devices optimally
4. Package and serve model stages
5. Load stages on each agent

### Step 6: Run Inference

```bash
# Test the pipeline
./manage.sh test

# Or run full inference
python run_inference.py \
  --plan plan.json \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --temperature 0.7
```

## Troubleshooting

### Android Agent Not Starting

1. **Check SSH connection:**
   ```bash
   ssh -p 8022 u0_a123@172.20.10.3 "echo 'Connected'"
   ```

2. **Check if agent is running on Android:**
   ```bash
   ssh -p 8022 u0_a123@172.20.10.3 "cd ~/LLM_project && cat .android_agent.log"
   ```

3. **Manually start on Android (in Termux):**
   ```bash
   cd ~/LLM_project
   source .venv/bin/activate
   ./start_android_agent.sh
   ```

### Android IP Changed

If your Android phone gets a new IP address:

1. Update `manage.sh`:
   ```bash
   # Edit line 16 in manage.sh
   ANDROID_IP="${ANDROID_IP:-172.20.10.3}"  # Change to new IP
   ```

2. Update `sync_to_android.sh`:
   ```bash
   # Edit line 8 in sync_to_android.sh
   ANDROID_IP="${1:-${ANDROID_IP:-172.20.10.3}}"  # Change to new IP
   ```

### Memory Issues on Android

Android phones typically have less RAM. If you get OOM errors:

1. Reduce memory fraction:
   ```bash
   --mem-fraction 0.35  # Instead of 0.45
   ```

2. Close other apps on your phone

3. The coordinator will automatically assign fewer layers to Android if needed

## Configuration Summary

- **PC Agent**: `http://127.0.0.1:8008`
- **Pi Agent**: `http://172.20.10.2:8008`
- **Android Agent**: `http://172.20.10.3:8008`
- **File Server**: `http://172.20.10.4:8090` (PC's network IP)

All IPs are already configured in the scripts. Just run the commands above!

