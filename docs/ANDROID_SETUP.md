# Android Phone Setup Guide

This guide will help you set up your Android phone as an agent in the distributed LLM inference system.

## Prerequisites

1. **Termux** - A terminal emulator for Android
   - Download from [F-Droid](https://f-droid.org/packages/com.termux/) (recommended) or Google Play Store
   - Note: Google Play version may be outdated, F-Droid is preferred

2. **Network Access**
   - Your Android phone must be on the same network as your PC and Raspberry Pi
   - You'll need the phone's IP address (check in WiFi settings)

3. **Storage Space**
   - At least 2-3GB free space for Python environment and model stages

## Step 1: Install Termux and Basic Tools

1. Open Termux
2. Update packages:
   ```bash
   pkg update && pkg upgrade -y
   ```

3. Install required packages:
   ```bash
   pkg install -y python git curl wget rsync openssh
   ```

## Step 2: Setup Python Environment

1. Create project directory:
   ```bash
   mkdir -p ~/LLM_project
   cd ~/LLM_project
   ```

2. Create virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Upgrade pip:
   ```bash
   pip install --upgrade pip
   ```

## Step 3: Install Python Dependencies

1. Install agent requirements:
   ```bash
   pip install fastapi==0.115.2 uvicorn==0.32.0 httpx==0.27.2 psutil==6.1.0 numpy==2.1.3 pydantic==2.9.2 safetensors==0.4.5
   ```

   **Note**: PyTorch is optional but recommended for accurate profiling. If you want to install it:
   ```bash
   # For ARM64 Android (most modern phones)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Step 4: Get Your Phone's IP Address

1. In Termux, run:
   ```bash
   ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d/ -f1
   ```

   Or check in your phone's WiFi settings → Advanced → IP address

2. **Note this IP address** - you'll need it for the coordinator

## Step 5: Sync Code from PC

From your PC, run:
```bash
./sync_to_android.sh
```

This will sync all necessary agent files to your Android phone.

## Step 6: Start the Android Agent

**Option A: From PC (recommended)**
```bash
./manage.sh start-android
```

**Option B: From Android phone (in Termux)**
```bash
cd ~/LLM_project
source .venv/bin/activate
./start_android_agent.sh
```

## Step 7: Verify Agent is Running

From your PC:
```bash
curl http://<ANDROID_IP>:8008/v1/health
```

You should see a JSON response with agent information.

## Step 8: Run Coordinator with Android

When running the coordinator, include your Android phone in the URLs and pipeline order:

```bash
python -m ebp.coordinator_main \
  --model-path /path/to/model \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008,http://172.20.10.3:8008" \
  --pipeline-order "pc,pi,android" \
  --min-prefix 4 \
  --mem-fraction 0.45 \
  --package
```

## Troubleshooting

### Port Already in Use
If port 8008 is already in use:
```bash
# Check what's using it
lsof -i :8008
# Or use a different port
python -m ebp.agent_main --name android --port 8009
```

### Network Connection Issues
1. Ensure phone and PC/Pi are on the same WiFi network
2. Check firewall settings on your phone
3. Verify IP address is correct
4. Try pinging from PC: `ping <ANDROID_IP>`

### Termux Background Execution
Termux may kill processes when the app is closed. To keep the agent running:
1. Use `termux-wake-lock` to prevent sleep
2. Or use a background task manager
3. Or run in a screen/tmux session

### Memory Issues
Android phones typically have less RAM than PCs. If you encounter OOM errors:
- Reduce `--mem-fraction` (try 0.3 or 0.35)
- Ensure no other heavy apps are running
- Consider assigning fewer layers to the Android device

## Notes

- **Battery**: Running inference will drain battery quickly. Keep phone plugged in.
- **Performance**: Android phones are typically slower than PCs/Pi. Expect longer inference times.
- **Storage**: Model stages can be large. Ensure sufficient storage space.
- **Network**: Use WiFi, not mobile data, to avoid data charges.

