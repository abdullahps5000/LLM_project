# Fix PyTorch Installation on Android

## Problem
PyTorch doesn't have official wheels for Android/ARM architecture used by Termux.

## Solutions

### Option 1: Try PyPI Direct Install (Recommended First Try)

```bash
# On Android in Termux:
cd ~/LLM_project
source .venv/bin/activate

# Try installing from PyPI directly (may have ARM builds)
pip install torch --no-cache-dir

# If that fails, try with specific version
pip install torch==2.1.0 --no-cache-dir
```

### Option 2: Check Your Architecture

```bash
# Check what architecture you have
python --version
python -c "import platform; print(f'Arch: {platform.machine()}')"
python -c "import sys; print(f'Platform: {sys.platform}')"
```

### Option 3: Try ARM64 Wheel (if your phone is ARM64)

```bash
# Most modern Android phones are ARM64
pip install torch --index-url https://download.pytorch.org/whl/cpu --platform linux_aarch64 --only-binary :all:
```

### Option 4: Install Without Torch (Temporary Workaround)

If PyTorch installation fails, we can modify the agent to work without torch for basic operations, but **forward passes will not work**. This means Android can participate in the pipeline but won't be able to run inference.

**This is a temporary solution - you'll need PyTorch for actual inference.**

### Option 5: Use Pre-built Termux Package (if available)

```bash
# Check if Termux has a PyTorch package
pkg search torch
# If available:
pkg install python-torch  # or similar
```

### Option 6: Build from Source (Advanced - Not Recommended)

Building PyTorch from source on Android is very difficult and time-consuming. Not recommended unless you have specific requirements.

## Recommended Approach

1. **First, try Option 1** (PyPI direct install)
2. **If that fails, check your architecture** (Option 2)
3. **If ARM64, try Option 3**
4. **If all fail, use Option 4** (workaround without torch) - but note that inference won't work on Android

## Alternative: Run Without Android

If PyTorch installation continues to fail, you can run the system with just PC + Pi:

```bash
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --min-prefix 4 \
  --mem-fraction 0.40 \
  --package
```

This will work perfectly fine - PC and Pi can handle all 28 layers efficiently.

