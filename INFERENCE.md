# Running Inference

This guide shows you how to run distributed LLM inference on your system.

## Prerequisites

1. **Services Running**: Make sure all agents and the file server are running
   ```bash
   ./scripts/manage.sh status
   ```

2. **Plan File**: You need a `plan.json` file from running the coordinator
   ```bash
   python -m ebp.coordinator_main --model-path /path/to/model --urls "..." --package
   ```

## Quick Start

### Basic Inference

```bash
source .venv/bin/activate
python run_inference.py --plan plan.json --prompt "Your prompt here" --max-tokens 50
```

### With Options

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "The weather today is" \
  --max-tokens 100 \
  --temperature 0.7 \
  --top-p 0.9
```

### Greedy Decoding (No Sampling)

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --greedy
```

## Parameters

- `--plan`: Path to the plan.json file (required)
- `--prompt`: Input text prompt (required)
- `--max-tokens`: Maximum number of tokens to generate (default: 50)
- `--temperature`: Sampling temperature (0.0-2.0, default: 1.0)
  - Lower = more deterministic
  - Higher = more creative/random
- `--top-p`: Nucleus sampling threshold (0.0-1.0, default: 0.9)
  - Only consider tokens with cumulative probability up to this value
- `--greedy`: Use greedy decoding (always pick highest probability token)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, default: INFO)

## Examples

### Creative Writing
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Once upon a time, in a magical forest," \
  --max-tokens 200 \
  --temperature 0.9 \
  --top-p 0.95
```

### Technical Question
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Explain how neural networks work:" \
  --max-tokens 150 \
  --temperature 0.3 \
  --greedy
```

### Conversation
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Hello! How can I help you today?" \
  --max-tokens 100 \
  --temperature 0.7
```

## Troubleshooting

### "Stage not loaded" Error
The stages are automatically loaded on first inference. If you get this error:
1. Check that agents are running: `./scripts/manage.sh status`
2. Check that the file server is running (port 8090)
3. Verify the plan.json references the correct stage directory

### Slow Generation
- First forward pass is slow (lazy model loading)
- Subsequent tokens are faster
- Large models or slow devices (Pi) will be slower

### Poor Quality Output
- Try lower temperature (0.3-0.7) for more coherent text
- Try greedy decoding for deterministic results
- Increase max-tokens if output is cut off

## How It Works

1. **Tokenization**: Your prompt is converted to token IDs
2. **Stage 0 (First Device)**: Processes input_ids through embeddings and first layers
3. **Intermediate Stages**: Hidden states flow through middle layers on other devices
4. **Final Stage**: Last layers produce logits, which are converted to tokens
5. **Generation Loop**: Repeats for each new token until max-tokens is reached

The system automatically handles:
- Loading stages on agents (lazy loading on first use)
- Passing hidden states between devices
- Tokenization and detokenization
- Sampling or greedy decoding

