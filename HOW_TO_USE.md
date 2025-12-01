# How to Use the Enhanced System

## 🚀 Quick Start

### 1. Install Dependencies

```bash
source .venv/bin/activate
pip install -r requirements-pc.txt
```

On Pi:
```bash
ssh abdoulaye@172.20.10.2
cd ~/LLM_project
source .venv/bin/activate
pip install -r requirements-agent.txt
```

### 2. Sync Code to Pi

```bash
./scripts/sync_to_pi.sh
```

### 3. Start Services

```bash
# PC
./scripts/manage.sh start

# Pi
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'

# Verify
./scripts/manage.sh status
```

### 4. Run Inference

**Option A: Unified Script (Recommended)**
```bash
source .venv/bin/activate

python run.py \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50 \
  --stream
```

**Option B: Separate Steps**
```bash
# 1. Run coordinator (if plan.json doesn't exist)
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50 \
  --package

# 2. Run inference
python run_inference.py \
  --plan plan.json \
  --prompt "Hello!" \
  --max-tokens 50 \
  --stream
```

## ✨ New Features

### Real-time Progress
- Progress bar shows tokens/sec
- `--stream` option streams tokens as they generate
- Per-stage timing in logs

### KV Cache (Much Faster!)
- Automatic incremental decoding
- 10-100x speedup for longer sequences
- Sessions manage KV cache automatically

### Binary Protocol
- Faster tensor transfer (default)
- Use `--no-binary` to disable if needed

### Security (Optional)
```bash
export EBP_AUTH_TOKEN="your-token"
export EBP_REQUIRE_AUTH="true"
# Restart agents
```

### Metrics
```bash
curl http://127.0.0.1:8008/v1/metrics
```

## 📊 What You'll See

**With progress bar:**
```
Generating: 100%|████████████| 50/50 [00:15<00:00, 3.2 tok/s, token=world]
```

**With streaming:**
```
Generating (streaming): Hello, how are you? I'm doing well, thank you for asking...
```

**In logs:**
```
[INFO] Prefill complete: 234.5ms
[INFO] Step 1/50: 'Hello' (12.3 tok/s)
[INFO] Per-stage average timing:
[INFO]   pc: 45.2ms avg
[INFO]   pi: 67.8ms avg
```

## 🔧 Troubleshooting

### Memory Error
```bash
# Increase memory fraction
--mem-fraction 0.50  # or 0.60
```

### Slow Generation
- Check metrics: `curl http://127.0.0.1:8008/v1/metrics`
- Verify KV cache is working (cache_hit_rate > 0)
- Check network latency

### "Session not found"
- Sessions are auto-created, this shouldn't happen
- Check agent logs if it persists

## 📚 More Info

See [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) for detailed documentation.

