# Running Without Android (Recommended)

## Why Run Without Android?

1. **PyTorch doesn't support Android/ARM** - No official wheels available
2. **Android is very slow** - 1.0 GFLOPS vs PC's 62.76 GFLOPS (62x slower!)
3. **PC + Pi is sufficient** - Together they have plenty of memory and compute
4. **Better performance** - No slow bottleneck in the pipeline

## How to Run

### Step 1: Run Coordinator with PC + Pi Only

```bash
python -m ebp.coordinator_main \
  --model-path /home/abdoulaye/myenv/my_models/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --min-prefix 4 \
  --mem-fraction 0.40 \
  --package
```

This will:
- Discover PC and Pi agents
- Partition model optimally across 2 devices
- Package 2 stages (instead of 3)
- Load stages on PC and Pi
- Complete successfully!

### Step 2: Run Inference

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --temperature 0.7
```

## Expected Partition

With PC + Pi, you'll typically get:
- **PC**: ~14-16 layers (handles first part of model)
- **Pi**: ~12-14 layers (handles second part of model)

This is well-balanced and will run efficiently!

## Performance Comparison

| Setup | Devices | Speed | Complexity |
|-------|---------|-------|------------|
| PC + Pi | 2 | Fast | Simple |
| PC + Pi + Android | 3 | Slow (Android bottleneck) | Complex |

**Recommendation**: Use PC + Pi only for best performance!

