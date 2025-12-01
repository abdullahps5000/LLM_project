# Android Phone Quick Start

## Quick Setup (5 minutes)

### 1. On Android Phone (Termux)

```bash
# Install Termux from F-Droid, then:
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

# Setup SSH (for syncing from PC)
sshd
passwd  # Set a password
whoami   # Note your username (usually u0_aXXX)

# Get your IP address
ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
```

### 2. On PC

```bash
# Android IP is already set in manage.sh (172.20.10.3)
# If you need to override, set:
# export ANDROID_IP=172.20.10.3
export ANDROID_USER=u0_a123  # Replace with your Termux username

# Sync code to Android
./sync_to_android.sh

# Start Android agent
./manage.sh start-android
```

### 3. Run Coordinator with Android

```bash
# Include Android in the pipeline
python -m ebp.coordinator_main \
  --model-path /path/to/model \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008,http://172.20.10.3:8008" \
  --pipeline-order "pc,pi,android" \
  --min-prefix 4 \
  --mem-fraction 0.45 \
  --package
```

## Common Commands

```bash
# Check status (includes Android if ANDROID_IP is set)
./manage.sh status

# Start Android agent
./manage.sh start-android

# Stop Android agent
./manage.sh stop-android

# Sync code updates to Android
./sync_to_android.sh
```

## Troubleshooting

**SSH Connection Failed?**
- Ensure `sshd` is running in Termux
- Check IP address is correct
- Verify phone and PC are on same WiFi network
- Try: `ssh -p 8022 $ANDROID_USER@$ANDROID_IP`

**Agent Not Responding?**
- Check if agent is running: `./manage.sh status`
- View logs on Android: `tail -f ~/LLM_project/.android_agent.log`
- Restart agent: `./manage.sh stop-android && ./manage.sh start-android`

**Memory Issues?**
- Android phones have less RAM. Try lower `--mem-fraction` (0.3-0.35)
- Close other apps on phone
- Assign fewer layers to Android device

## Notes

- Keep phone plugged in (battery drain is high)
- Use WiFi, not mobile data
- Termux may kill processes when app closes - use `termux-wake-lock` to prevent sleep

