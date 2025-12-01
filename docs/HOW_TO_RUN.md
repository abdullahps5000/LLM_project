# How to Run Distributed Inference - Simplified

## Quick Start (One Command!)

### Start Everything
```bash
cd ~/LLM_project
source .venv/bin/activate

# Start PC services (Agent + File Server)
./manage.sh start

# Start Pi Agent (one-time setup, then it runs in background)
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
```

### Run Test
```bash
./manage.sh test
```

That's it! The `manage.sh test` command will:
- Check if services are running
- Auto-start PC services if needed
- Run the inference test

## Management Commands

### Check Status
```bash
./manage.sh status
```
Shows status of all services (PC Agent, File Server, Pi Agent)

### Stop Services
```bash
./manage.sh stop
```
Stops PC Agent and File Server

### Restart Services
```bash
./manage.sh restart
```

## Pi Agent Management

### Start Pi Agent
```bash
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
```

### Stop Pi Agent
```bash
ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./stop_pi_agent.sh'
```

## What Each Service Does

- **PC Agent**: Runs on your PC, handles Stage 0
- **Pi Agent**: Runs on Raspberry Pi, handles Stage 1  
- **File Server**: Serves stage files to Pi (runs on PC, port 8090)

## Workflow

1. **First Time Setup:**
   ```bash
   # Sync code to Pi
   ./sync_to_pi.sh
   
   # Start Pi Agent (runs in background)
   ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
   ```

2. **Daily Use:**
   ```bash
   # Start PC services
   ./manage.sh start
   
   # Run tests
   ./manage.sh test
   
   # Check status anytime
   ./manage.sh status
   
   # Stop when done
   ./manage.sh stop
   ```

## Troubleshooting

- **"Port already in use"**: Run `./manage.sh stop` first
- **"Pi Agent not running"**: Start it with `ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'`
- **Check logs**: 
  - PC Agent: `.pc_agent.log`
  - File Server: `.file_server.log`
  - Pi Agent: `.pi_agent.log` (on Pi)

## Advanced: Manual Control

If you prefer manual control, you can still run services individually:

```bash
# PC Agent
python -m ebp.agent_main --name pc --port 8008

# File Server
cd stages_out && python -m http.server 8090

# Test
python test_inference.py plan.json
```

But the unified manager (`manage.sh`) is much easier! 🚀
