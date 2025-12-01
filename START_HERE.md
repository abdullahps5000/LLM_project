# 🚀 START HERE - How to Run Inference

## Quick Answer to Your Question

**Q: Why are coordinator and inference separate? Can I run inference without coordinator?**

**A:** Yes! You can run inference without the coordinator IF you already have a `plan.json` file.

- **Coordinator**: Creates the partitioning plan (`plan.json`) - only needed once or when you change models/devices
- **Inference**: Uses the plan to generate text - can run many times with the same plan

## 🎯 Easiest Way to Run (Recommended)

Use the new unified `run.py` script - it automatically runs coordinator if needed:

```bash
source .venv/bin/activate

python run.py \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50
```

**If you already have `plan.json`, just:**
```bash
python run.py --prompt "Your prompt" --max-tokens 50
```

## 📋 Complete Step-by-Step

### Step 1: Start Services

```bash
# Start PC Agent and File Server
./scripts/manage.sh start

# Start Pi Agent
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'

# Verify everything is running
./scripts/manage.sh status
```

### Step 2: Run Inference

**Option A: Unified Script (Auto-runs coordinator if needed)**
```bash
source .venv/bin/activate
python run.py \
  --prompt "Hello!" \
  --max-tokens 30 \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50
```

**Option B: Separate Steps (More Control)**
```bash
# 1. Run coordinator (once)
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50 \
  --package

# 2. Run inference (many times)
python run_inference.py \
  --plan plan.json \
  --prompt "Your prompt" \
  --max-tokens 50 \
  --temperature 0.7
```

## 🔧 Fixing Memory Errors

If you see "Model too large" error:

1. **Increase memory fraction:**
   ```bash
   --mem-fraction 0.50  # or 0.60, 0.70
   ```

2. **Free up RAM:**
   - Close other applications
   - Check: `free -h`
   - Restart: `./scripts/manage.sh restart`

3. **Check device memory:**
   ```bash
   ./scripts/manage.sh status
   ```

## 📚 More Information

- **[HOW_TO_RUN.md](HOW_TO_RUN.md)** - Detailed explanation
- **[INFERENCE.md](INFERENCE.md)** - Inference options and examples
- **[README.md](README.md)** - Full project documentation

## ✨ What's New

- ✅ **Unified `run.py` script** - Automatically handles coordinator if needed
- ✅ **Better error messages** - Clear suggestions when memory is insufficient
- ✅ **Improved memory handling** - Better detection and suggestions
- ✅ **Cleaner project structure** - Everything organized

## 🎉 You're Ready!

Just run:
```bash
python run.py --prompt "Your prompt" --max-tokens 50
```

If `plan.json` exists, it will run inference. If not, it will run coordinator first!

