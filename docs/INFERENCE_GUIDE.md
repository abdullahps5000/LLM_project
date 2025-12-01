# Full Inference Guide

## Quick Start

### 1. Start Services
```bash
./manage.sh start
```

### 2. Start Pi Agent (if not running)
```bash
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
```

### 3. Run Inference
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Once upon a time, in a magical land," \
  --max-tokens 100 \
  --temperature 0.7
```

## Options

- `--plan`: Path to plan.json (required)
- `--prompt`: Input text prompt (required)
- `--max-tokens`: Maximum tokens to generate (default: 50)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-p`: Top-p (nucleus) sampling (default: 0.9)
- `--greedy`: Use greedy decoding instead of sampling
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR)

## Examples

### Creative Writing
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "In a distant galaxy, explorers discovered" \
  --max-tokens 150 \
  --temperature 0.8
```

### Technical Explanation
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Explain how neural networks work:" \
  --max-tokens 200 \
  --temperature 0.5 \
  --greedy
```

### Conversation
```bash
python run_inference.py \
  --plan plan.json \
  --prompt "User: What is machine learning?\nAssistant:" \
  --max-tokens 100 \
  --temperature 0.7
```

## How It Works

1. **Tokenization**: Input text is converted to token IDs
2. **Pipeline Forward**: Tokens flow through each stage:
   - Stage 0 (PC): Embeddings + Layers 0-16
   - Stage 1 (Pi): Layers 17-27 + LM Head
3. **Generation Loop**: 
   - Sample next token from logits
   - Append to sequence
   - Repeat until max_tokens or EOS
4. **Decoding**: Convert token IDs back to text

## Troubleshooting

- **"Model not loaded"**: Models load lazily on first forward pass. First generation will be slower.
- **"Connection refused"**: Check that all agents are running with `./manage.sh status`
- **"Out of memory"**: Reduce model size or adjust memory settings
- **Slow generation**: Normal for first run (model loading). Subsequent runs are faster.

## Performance Tips

- First generation: ~10-30 seconds (model loading)
- Subsequent generations: ~1-5 seconds per token
- Use `--greedy` for faster (but less creative) output
- Lower `--temperature` for more focused output

