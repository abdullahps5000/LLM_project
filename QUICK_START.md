# Quick Start Guide

## Step 1: Start Services

```bash
# Start PC Agent and File Server
./scripts/manage.sh start

# Start Pi Agent (on Pi or via SSH)
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'

# Check all services are running
./scripts/manage.sh status
```

## Step 2: Run Coordinator

```bash
source .venv/bin/activate

python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.40 \
  --package
```

This creates `plan.json` and packages stages.

## Step 3: Run Inference

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --temperature 0.7
```

## That's It!

See [INFERENCE.md](INFERENCE.md) for more inference options.

