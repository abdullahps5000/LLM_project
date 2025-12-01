#!/bin/bash
# EBP Service Manager - Unified script to manage all services
# Usage: ./manage.sh [start|stop|status|test|restart]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
PC_AGENT_PORT=8008
FILE_SERVER_PORT=8090
PI_IP="${PI_IP:-172.20.10.2}"
PI_USER="${PI_USER:-abdoulaye}"
PI_PATH="${PI_PATH:-/home/$PI_USER/LLM_project}"
ANDROID_IP="${ANDROID_IP:-172.20.10.3}"
ANDROID_USER="${ANDROID_USER:-u0_a123}"
ANDROID_PATH="${ANDROID_PATH:-/data/data/com.termux/files/home/LLM_project}"
ANDROID_PORT="${ANDROID_PORT:-8022}"

# PID files
PC_AGENT_PID="$PROJECT_ROOT/.pc_agent.pid"
FILE_SERVER_PID="$PROJECT_ROOT/.file_server.pid"
PC_AGENT_LOG="$PROJECT_ROOT/.pc_agent.log"
FILE_SERVER_LOG="$PROJECT_ROOT/.file_server.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠ Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

start_pc_agent() {
    if [ -f "$PC_AGENT_PID" ] && kill -0 "$(cat "$PC_AGENT_PID")" 2>/dev/null; then
        echo -e "${YELLOW}PC Agent already running (PID: $(cat "$PC_AGENT_PID"))${NC}"
        # Wait a bit and verify it's actually responding
        sleep 1
        if curl -s --connect-timeout 2 "http://127.0.0.1:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ PC Agent is responding${NC}"
        else
            echo -e "${YELLOW}⚠ PC Agent process exists but not responding yet...${NC}"
        fi
        return 0
    fi
    
    echo -e "${GREEN}Starting PC Agent on port $PC_AGENT_PORT...${NC}"
    nohup python -m ebp.agent_main --name pc --port "$PC_AGENT_PORT" > "$PC_AGENT_LOG" 2>&1 &
    echo $! > "$PC_AGENT_PID"
    sleep 3
    
    if kill -0 "$(cat "$PC_AGENT_PID")" 2>/dev/null; then
        echo -e "${GREEN}✓ PC Agent started (PID: $(cat "$PC_AGENT_PID"))${NC}"
        echo -e "${YELLOW}Waiting for agent to be ready...${NC}"
        # Wait up to 5 seconds for agent to be ready
        for i in {1..10}; do
            if curl -s --connect-timeout 2 "http://127.0.0.1:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ PC Agent is ready${NC}"
                return 0
            fi
            sleep 0.5
        done
        echo -e "${YELLOW}⚠ PC Agent started but not responding yet (may need a moment)${NC}"
        return 0
    else
        echo -e "${RED}✗ PC Agent failed to start. Check $PC_AGENT_LOG${NC}"
        rm -f "$PC_AGENT_PID"
        return 1
    fi
}

stop_pc_agent() {
    if [ ! -f "$PC_AGENT_PID" ]; then
        echo -e "${YELLOW}PC Agent not running${NC}"
        return 0
    fi
    
    PID=$(cat "$PC_AGENT_PID")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}Stopping PC Agent (PID: $PID)...${NC}"
        kill "$PID" 2>/dev/null || true
        sleep 1
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$PC_AGENT_PID"
    echo -e "${GREEN}✓ PC Agent stopped${NC}"
}

start_file_server() {
    if [ -f "$FILE_SERVER_PID" ] && kill -0 "$(cat "$FILE_SERVER_PID")" 2>/dev/null; then
        echo -e "${YELLOW}File Server already running (PID: $(cat "$FILE_SERVER_PID"))${NC}"
        return 0
    fi
    
    # Check if port is already in use
    EXISTING_PID=$(lsof -ti:"$FILE_SERVER_PORT" 2>/dev/null | head -1)
    if [ -n "$EXISTING_PID" ]; then
        echo -e "${YELLOW}⚠ Port $FILE_SERVER_PORT already in use (PID: $EXISTING_PID).${NC}"
        echo -e "${GREEN}✓ File Server appears to be running${NC}"
        echo "$EXISTING_PID" > "$FILE_SERVER_PID"
        return 0
    fi
    
    # Find latest stage directory
    STAGES_DIR="$PROJECT_ROOT/stages_out"
    if [ ! -d "$STAGES_DIR" ]; then
        echo -e "${YELLOW}⚠ No stages_out directory found. File server not needed yet.${NC}"
        return 0
    fi
    
    echo -e "${GREEN}Starting File Server on port $FILE_SERVER_PORT...${NC}"
    cd "$STAGES_DIR"
    nohup python3 -m http.server "$FILE_SERVER_PORT" > "$FILE_SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > "$FILE_SERVER_PID"
    cd "$PROJECT_ROOT"
    sleep 2
    
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${GREEN}✓ File Server started (PID: $SERVER_PID)${NC}"
        return 0
    else
        echo -e "${RED}✗ File Server failed to start. Check $FILE_SERVER_LOG${NC}"
        cat "$FILE_SERVER_LOG" 2>/dev/null | tail -5
        rm -f "$FILE_SERVER_PID"
        return 1
    fi
}

stop_file_server() {
    if [ ! -f "$FILE_SERVER_PID" ]; then
        echo -e "${YELLOW}File Server not running${NC}"
        return 0
    fi
    
    PID=$(cat "$FILE_SERVER_PID")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${GREEN}Stopping File Server (PID: $PID)...${NC}"
        kill "$PID" 2>/dev/null || true
        sleep 1
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
    fi
    rm -f "$FILE_SERVER_PID"
    echo -e "${GREEN}✓ File Server stopped${NC}"
}

check_service() {
    local name=$1
    local url=$2
    local max_attempts=${3:-3}
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --connect-timeout 2 "$url/v1/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is running${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 0.5
    done
    
    echo -e "${RED}✗ $name is not responding${NC}"
    return 1
}

status() {
    echo "=========================================="
    echo "EBP Service Status"
    echo "=========================================="
    echo ""
    
    # PC Agent
    if [ -f "$PC_AGENT_PID" ] && kill -0 "$(cat "$PC_AGENT_PID")" 2>/dev/null; then
        echo -e "${GREEN}PC Agent:${NC} Running (PID: $(cat "$PC_AGENT_PID"))"
        check_service "PC Agent" "http://127.0.0.1:$PC_AGENT_PORT"
    else
        echo -e "${RED}PC Agent:${NC} Not running"
    fi
    echo ""
    
    # File Server
    if [ -f "$FILE_SERVER_PID" ] && kill -0 "$(cat "$FILE_SERVER_PID")" 2>/dev/null; then
        echo -e "${GREEN}File Server:${NC} Running (PID: $(cat "$FILE_SERVER_PID"))"
        if curl -s "http://127.0.0.1:$FILE_SERVER_PORT" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ File Server is responding${NC}"
        else
            echo -e "${RED}✗ File Server is not responding${NC}"
        fi
    else
        echo -e "${RED}File Server:${NC} Not running"
    fi
    echo ""
    
    # Pi Agent
    echo -e "${YELLOW}Pi Agent:${NC} Checking $PI_USER@$PI_IP:$PI_PATH"
    if ssh -o ConnectTimeout=2 -o BatchMode=yes "$PI_USER@$PI_IP" "cd $PI_PATH && test -f .pi_agent.pid && kill -0 \$(cat .pi_agent.pid) 2>/dev/null" 2>/dev/null; then
        echo -e "${GREEN}✓ Pi Agent process is running${NC}"
        check_service "Pi Agent" "http://$PI_IP:$PC_AGENT_PORT"
    else
        echo -e "${RED}✗ Pi Agent is not running${NC}"
        echo -e "${YELLOW}  Start it with: ssh $PI_USER@$PI_IP 'cd $PI_PATH && ./start_pi_agent.sh'${NC}"
    fi
    echo ""
    
    if [ -n "$ANDROID_IP" ]; then
        echo -e "${YELLOW}Android Agent:${NC} Checking $ANDROID_USER@$ANDROID_IP:$ANDROID_PATH"
        if ssh -p "$ANDROID_PORT" -o ConnectTimeout=2 -o BatchMode=yes "$ANDROID_USER@$ANDROID_IP" "cd $ANDROID_PATH && test -f .android_agent.pid && kill -0 \$(cat .android_agent.pid) 2>/dev/null" 2>/dev/null; then
            echo -e "${GREEN}✓ Android Agent process is running${NC}"
            check_service "Android Agent" "http://$ANDROID_IP:$PC_AGENT_PORT"
        else
            echo -e "${RED}✗ Android Agent is not running${NC}"
            echo -e "${YELLOW}  Start it with: ./manage.sh start-android${NC}"
        fi
        echo ""
    fi
    
    echo "=========================================="
}

start() {
    echo "=========================================="
    echo "Starting EBP Services"
    echo "=========================================="
    echo ""
    
    start_pc_agent
    echo ""
    start_file_server
    echo ""
    
    echo "=========================================="
    echo -e "${GREEN}Services started!${NC}"
    echo ""
    echo "Note: Remote agents must be started separately:"
    echo "  Pi:     ssh $PI_USER@$PI_IP 'cd $PI_PATH && ./start_pi_agent.sh'"
    if [ -n "$ANDROID_IP" ]; then
        echo "  Android: ./manage.sh start-android"
    fi
    echo ""
    echo "Check status with: ./manage.sh status"
    echo "=========================================="
}

stop() {
    echo "=========================================="
    echo "Stopping EBP Services"
    echo "=========================================="
    echo ""
    
    stop_pc_agent
    echo ""
    stop_file_server
    echo ""
    
    echo "=========================================="
    echo -e "${GREEN}Services stopped!${NC}"
    echo "=========================================="
}

restart() {
    stop
    sleep 2
    start
}

test_inference() {
    echo "=========================================="
    echo "Running Inference Test"
    echo "=========================================="
    echo ""
    
    # Check if services are running and wait for them to be ready
    echo -e "${YELLOW}Checking services...${NC}"
    
    PC_READY=false
    for i in {1..10}; do
        if curl -s --connect-timeout 2 "http://127.0.0.1:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
            PC_READY=true
            break
        fi
        sleep 0.5
    done
    
    if [ "$PC_READY" = false ]; then
        echo -e "${RED}✗ PC Agent is not responding${NC}"
        echo -e "${YELLOW}Starting PC Agent...${NC}"
        start_pc_agent
        echo -e "${YELLOW}Waiting for PC Agent to be ready...${NC}"
        for i in {1..15}; do
            if curl -s --connect-timeout 2 "http://127.0.0.1:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
                PC_READY=true
                break
            fi
            sleep 0.5
        done
        if [ "$PC_READY" = false ]; then
            echo -e "${RED}✗ PC Agent failed to start. Check $PC_AGENT_LOG${NC}"
            exit 1
        fi
    fi
    echo -e "${GREEN}✓ PC Agent is ready${NC}"
    
    PI_READY=false
    for i in {1..5}; do
        if curl -s --connect-timeout 2 "http://$PI_IP:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
            PI_READY=true
            break
        fi
        sleep 0.5
    done
    
    if [ "$PI_READY" = false ]; then
        echo -e "${RED}✗ Pi Agent is not responding${NC}"
        echo -e "${YELLOW}Please start Pi Agent first:${NC}"
        echo "  ssh $PI_USER@$PI_IP 'cd $PI_PATH && ./start_pi_agent.sh'"
        exit 1
    fi
    echo -e "${GREEN}✓ Pi Agent is ready${NC}"
    
    if [ -n "$ANDROID_IP" ]; then
        ANDROID_READY=false
        for i in {1..5}; do
            if curl -s --connect-timeout 2 "http://$ANDROID_IP:$PC_AGENT_PORT/v1/health" > /dev/null 2>&1; then
                ANDROID_READY=true
                break
            fi
            sleep 0.5
        done
        
        if [ "$ANDROID_READY" = false ]; then
            echo -e "${RED}✗ Android Agent is not responding${NC}"
            echo -e "${YELLOW}Please start Android Agent first:${NC}"
            echo "  ./manage.sh start-android"
            exit 1
        fi
        echo -e "${GREEN}✓ Android Agent is ready${NC}"
    fi
    
    if [ ! -f "plan.json" ]; then
        echo -e "${RED}✗ plan.json not found${NC}"
        echo "Run coordinator first to create plan.json"
        exit 1
    fi
    
    echo ""
    echo -e "${GREEN}Running test...${NC}"
    echo ""
    python test_inference.py plan.json
}

start_android() {
    if [ -z "$ANDROID_IP" ]; then
        echo -e "${RED}ERROR: ANDROID_IP not set${NC}"
        echo ""
        echo "Set it with: export ANDROID_IP=<your_android_ip>"
        echo "Or edit manage.sh to set default ANDROID_IP"
        exit 1
    fi
    
    echo "=========================================="
    echo "Starting Android Agent"
    echo "=========================================="
    echo ""
    
    echo -e "${YELLOW}Checking if Android Agent is already running...${NC}"
    if ssh -p "$ANDROID_PORT" -o ConnectTimeout=2 -o BatchMode=yes "$ANDROID_USER@$ANDROID_IP" "cd $ANDROID_PATH && test -f .android_agent.pid && kill -0 \$(cat .android_agent.pid) 2>/dev/null" 2>/dev/null; then
        echo -e "${YELLOW}Android Agent already running${NC}"
        exit 0
    fi
    
    echo -e "${GREEN}Starting Android Agent on $ANDROID_IP...${NC}"
    ssh -p "$ANDROID_PORT" "$ANDROID_USER@$ANDROID_IP" "cd $ANDROID_PATH && ./start_android_agent.sh"
    
    sleep 2
    
    if check_service "Android Agent" "http://$ANDROID_IP:$PC_AGENT_PORT" 2>/dev/null; then
        echo ""
        echo -e "${GREEN}✓ Android Agent started successfully${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠ Android Agent may still be starting. Check status with: ./manage.sh status${NC}"
    fi
    
    echo "=========================================="
}

stop_android() {
    if [ -z "$ANDROID_IP" ]; then
        echo -e "${RED}ERROR: ANDROID_IP not set${NC}"
        exit 1
    fi
    
    echo "=========================================="
    echo "Stopping Android Agent"
    echo "=========================================="
    echo ""
    
    echo -e "${GREEN}Stopping Android Agent on $ANDROID_IP...${NC}"
    ssh -p "$ANDROID_PORT" "$ANDROID_USER@$ANDROID_IP" "cd $ANDROID_PATH && ./stop_android_agent.sh" || echo -e "${YELLOW}Android Agent may not be running${NC}"
    
    echo ""
    echo -e "${GREEN}✓ Android Agent stopped${NC}"
    echo "=========================================="
}

case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    test)
        test_inference
        ;;
    start-android)
        start_android
        ;;
    stop-android)
        stop_android
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test|start-android|stop-android}"
        echo ""
        echo "Commands:"
        echo "  start         - Start PC Agent and File Server"
        echo "  stop          - Stop PC Agent and File Server"
        echo "  restart       - Restart PC Agent and File Server"
        echo "  status        - Show status of all services"
        echo "  test          - Run inference test (auto-starts services if needed)"
        echo "  start-android - Start Android Agent (requires ANDROID_IP)"
        echo "  stop-android  - Stop Android Agent (requires ANDROID_IP)"
        exit 1
        ;;
esac

