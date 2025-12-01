# Quick Start Guide

## One-Time Setup

1. **Sync code to Pi:**
   ```bash
   ./sync_to_pi.sh
   ```

2. **Start Pi Agent (runs in background):**
   ```bash
   ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'
   ```

## Daily Use (Just 2 Commands!)

### Start Everything
```bash
cd ~/LLM_project
source .venv/bin/activate
./manage.sh start
```

### Run Test
```bash
./manage.sh test
```

## All Commands

| Command | What It Does |
|---------|-------------|
| `./manage.sh start` | Start PC Agent + File Server |
| `./manage.sh stop` | Stop PC Agent + File Server |
| `./manage.sh status` | Check all services |
| `./manage.sh test` | Run inference test |
| `./manage.sh restart` | Restart PC services |

## Pi Agent Commands

| Command | What It Does |
|---------|-------------|
| `ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./start_pi_agent.sh'` | Start Pi Agent |
| `ssh abdoulaye@172.20.10.2 'cd ~/LLM_project && ./stop_pi_agent.sh'` | Stop Pi Agent |

## That's It!

No more multiple terminals! Everything runs in the background. 🎉

