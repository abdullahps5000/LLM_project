# EBP - Edge-Based Pipeline for Distributed LLM Inference

**Production-ready distributed LLM inference system with KV cache, binary protocol, and real-time progress tracking.**

## 🚀 Quick Start

### 1. Start Services

```bash
# PC: Start agent and file server
./scripts/manage.sh start

# Pi: Start agent
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'

# Verify
./scripts/manage.sh status
```

### 2. Run Inference (Unified - Recommended)

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

**If you already have `plan.json`:**
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Your prompt" \
  --max-tokens 50 \
  --stream
```

## ✨ Key Features

- ✅ **KV Cache** - 10-100x faster generation with incremental decoding
- ✅ **Binary Protocol** - 2-5x faster tensor transfer
- ✅ **Real-time Progress** - See tokens as they generate
- ✅ **Session Management** - Proper KV cache handling
- ✅ **Security** - Optional authentication and checksums
- ✅ **Validation** - Stage weight verification
- ✅ **Metrics** - Performance monitoring
- ✅ **Better Profiling** - Real layer timing benchmarks

## 📚 Documentation

- **[HOW_TO_USE.md](HOW_TO_USE.md)** - Quick start guide
- **[COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)** - Complete documentation
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - What changed
- **[INFERENCE.md](INFERENCE.md)** - Inference options

## 🎯 Performance

**Before:** O(T²) complexity, JSON serialization, no progress
**After:** O(T) complexity, binary protocol, real-time progress

**Speedup:** 10-100x for longer sequences

## 🔧 Requirements

### PC
```bash
pip install -r requirements-pc.txt
```

### Pi/Agents
```bash
pip install -r requirements-agent.txt
```

## 📊 Monitoring

```bash
# Check agent metrics
curl http://127.0.0.1:8008/v1/metrics
curl http://172.20.10.2:8008/v1/metrics
```

## 🔒 Security (Optional)

```bash
export EBP_AUTH_TOKEN="your-secret-token"
export EBP_REQUIRE_AUTH="true"
# Restart agents
```

## 🎉 What's New

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for complete list of improvements.

The system is now **production-ready** with:
- KV cache for fast generation
- Binary protocol for efficiency  
- Progress tracking
- Security options
- Better profiling
- Validation
- Monitoring

## 📖 More Info

- **Quick Start:** [HOW_TO_USE.md](HOW_TO_USE.md)
- **Complete Guide:** [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md)
- **Improvements:** [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)
