# How to Run Distributed LLM Inference

## Quick Answer

**You can run inference without the coordinator IF you already have a `plan.json` file.**

The coordinator creates the partitioning plan (`plan.json`). Once created, you can reuse it for multiple inference runs.

## Two Ways to Run

### Option 1: Unified Script (Recommended)

The new `run.py` script automatically runs the coordinator if needed:

```bash
source .venv/bin/activate

# If plan.json doesn't exist, it will run coordinator first
python run.py \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50
```

### Option 2: Separate Steps (More Control)

**Step 1: Run Coordinator (once, or when you want to repartition)**

```bash
source .venv/bin/activate
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50 \
  --package
```

**Step 2: Run Inference (can run many times with same plan.json)**

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Your prompt here" \
  --max-tokens 50 \
  --temperature 0.7
```

## Fixing Memory Issues

If you get "Model too large" error:

1. **Increase memory fraction:**
   ```bash
   --mem-fraction 0.50  # or 0.60, 0.70
   ```

2. **Free up RAM:**
   - Close other applications
   - Check available RAM: `free -h`
   - Restart services: `./scripts/manage.sh restart`

3. **Check device memory:**
   ```bash
   ./scripts/manage.sh status
   ```

## Complete Workflow

```bash
# 1. Start services
./scripts/manage.sh start
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
./scripts/manage.sh status

# 2. Run inference (coordinator runs automatically if needed)
python run.py \
  --prompt "Hello!" \
  --max-tokens 30 \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.50
```

## Why Separate?

- **Coordinator**: Heavy operation (profiles devices, partitions model, packages stages)
- **Inference**: Light operation (just runs forward passes)
- **Reusability**: Once partitioned, you can run inference many times without repartitioning
- **Flexibility**: You can manually adjust the plan or use different models

## Tips

- Keep `plan.json` - you can reuse it for multiple inference runs
- Only rerun coordinator if you change model, devices, or memory settings
- Use `--mem-fraction 0.50` or higher if you have enough RAM
- Check `./scripts/manage.sh status` to see available RAM on each device

