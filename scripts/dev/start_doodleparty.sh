#!/bin/bash
# DoodleParty Unified Startup Script
# Starts both Express server and ML inference server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "🎨 DOODLEPARTY - STARTING ALL SERVICES"
echo "========================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "../../frontend/server/express-server.cjs" ]; then
    echo -e "${RED}❌ Express server file not found${NC}"
    exit 1
fi

if [ ! -f "../../ml/socket_client/ml_server.py" ]; then
    echo -e "${RED}❌ ML server file not found${NC}"
    exit 1
fi

# Create log directory
mkdir -p logs

# PID files
EXPRESS_PID_FILE="logs/express.pid"
ML_PID_FILE="logs/ml.pid"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"
    
    # Kill Express server
    if [ -f "$EXPRESS_PID_FILE" ]; then
        EXPRESS_PID=$(cat "$EXPRESS_PID_FILE")
        if ps -p $EXPRESS_PID > /dev/null 2>&1; then
            echo -e "${CYAN}  Stopping Express server (PID: $EXPRESS_PID)${NC}"
            kill $EXPRESS_PID 2>/dev/null || true
        fi
        rm -f "$EXPRESS_PID_FILE"
    fi
    
    # Kill ML server
    if [ -f "$ML_PID_FILE" ]; then
        ML_PID=$(cat "$ML_PID_FILE")
        if ps -p $ML_PID > /dev/null 2>&1; then
            echo -e "${CYAN}  Stopping ML server (PID: $ML_PID)${NC}"
            kill $ML_PID 2>/dev/null || true
        fi
        rm -f "$ML_PID_FILE"
    fi
    
    echo -e "${GREEN}✓ All services stopped${NC}"
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

# Start Express server
echo -e "${BLUE}📦 Starting Express server...${NC}"
cd ../../frontend/server
node express-server.cjs > ../../logs/express.log 2>&1 &
EXPRESS_PID=$!
echo $EXPRESS_PID > "../../logs/$EXPRESS_PID_FILE"
cd ../../scripts/dev

# Wait for Express server to start
echo -e "${CYAN}  Waiting for Express server to initialize...${NC}"
sleep 2

# Check if Express server is running
if ps -p $EXPRESS_PID > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ Express server started (PID: $EXPRESS_PID)${NC}"
    echo -e "${CYAN}    URL: http://localhost:3000${NC}"
    echo -e "${CYAN}    DoodleParty: http://localhost:3000/doodleparty${NC}"
else
    echo -e "${RED}  ❌ Express server failed to start${NC}"
    echo -e "${YELLOW}  Check logs/express.log for details${NC}"
    exit 1
fi

echo ""

# Start ML server (Python)
echo -e "${BLUE}🤖 Starting ML inference server...${NC}"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo -e "${CYAN}  Activating virtual environment...${NC}"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo -e "${CYAN}  Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}  ⚠️  No virtual environment found, using system Python${NC}"
fi

# Check if required Python packages are installed
MISSING_PACKAGES=0
if ! python3 -c "import socketio" 2>/dev/null; then
    echo -e "${YELLOW}  ⚠️  python-socketio not installed${NC}"
    MISSING_PACKAGES=1
fi

if ! python3 -c "import websocket" 2>/dev/null; then
    echo -e "${YELLOW}  ⚠️  websocket-client not installed${NC}"
    MISSING_PACKAGES=1
fi

if [ $MISSING_PACKAGES -eq 1 ]; then
    echo -e "${CYAN}  Installing missing packages...${NC}"
    pip install -q python-socketio[client] websocket-client eventlet 2>&1 | grep -v "Requirement already satisfied" || true
    echo -e "${GREEN}  ✓ Packages installed${NC}"
fi

# Start ML server
python3 ../../ml/socket_client/ml_server.py > ../../logs/ml.log 2>&1 &
ML_PID=$!
echo $ML_PID > "../../logs/$ML_PID_FILE"

# Wait for ML server to start
echo -e "${CYAN}  Waiting for ML server to connect...${NC}"
sleep 3

# Check if ML server is running
if ps -p $ML_PID > /dev/null 2>&1; then
    echo -e "${GREEN}  ✓ ML server started (PID: $ML_PID)${NC}"
else
    echo -e "${RED}  ❌ ML server failed to start${NC}"
    echo -e "${YELLOW}  Check logs/ml.log for details${NC}"
    cleanup
    exit 1
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}✅ ALL SERVICES RUNNING${NC}"
echo "========================================================================"
echo ""
echo -e "${CYAN}Services:${NC}"
echo -e "  📦 Express Server:  http://localhost:3000"
echo -e "  🎨 DoodleParty:     http://localhost:3000/doodleparty"
echo -e "  🤖 ML Server:       Connected and listening"
echo ""
echo -e "${CYAN}Logs:${NC}"
echo -e "  Express: logs/express.log"
echo -e "  ML:      logs/ml.log"
echo ""
echo -e "${CYAN}Visualizations:${NC}"
echo -e "  Saved to: data/ml_visualizations/"
echo -e "  Detections: data/ml_detections/"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running and monitor processes
while true; do
    # Check if Express server is still running
    if ! ps -p $EXPRESS_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ Express server died unexpectedly${NC}"
        cleanup
        exit 1
    fi
    
    # Check if ML server is still running
    if ! ps -p $ML_PID > /dev/null 2>&1; then
        echo -e "${RED}❌ ML server died unexpectedly${NC}"
        cleanup
        exit 1
    fi
    
    sleep 5
done
