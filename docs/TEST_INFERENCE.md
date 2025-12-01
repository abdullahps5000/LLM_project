# Testing Full Inference

## Quick Test

```bash
# Make sure services are running
./manage.sh status

# Run inference
python run_inference.py \
  --plan plan.json \
  --prompt "Hello, how are you?" \
  --max-tokens 20 \
  --temperature 0.7
```

## Troubleshooting 422 Errors

If you get a 422 Unprocessable Entity error:

1. **Check agent logs:**
   ```bash
   tail -50 .pc_agent.log
   tail -50 .pi_agent.log  # on Pi
   ```

2. **Restart agents:**
   ```bash
   ./manage.sh restart
   ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./stop_pi_agent.sh && ./start_pi_agent.sh'
   ```

3. **Verify stage metadata:**
   - Make sure stages were loaded with model_path and layer_range
   - Check that plan.json has correct layer_ranges

4. **Test with simple prompt:**
   ```bash
   python run_inference.py --plan plan.json --prompt "Hi" --max-tokens 5
   ```

## Expected Behavior

- **First generation**: Slower (10-30s) - models load on first forward pass
- **Subsequent generations**: Faster (1-5s per token)
- **Output**: Generated text continuation

## Debug Mode

```bash
python run_inference.py \
  --plan plan.json \
  --prompt "Test" \
  --max-tokens 10 \
  --log-level DEBUG
```

