#!/bin/bash
# Start DoodleParty with ML Content Detection
#
# This script starts both:
# 1. ML Server (Flask) on port 5000
# 2. Express Canvas Server on port 3000

set -e  # Exit on error

# Configuration
ML_PORT=5000
EXPRESS_PORT=3000
ML_MODEL_PATH="${ML_MODEL_PATH:-$(pwd)/models/quickdraw_model_int8.tflite}"
ML_INPUT_SIZE="${ML_INPUT_SIZE:-128}"
ML_CONFIDENCE_THRESHOLD="${ML_CONFIDENCE_THRESHOLD:-0.7}"
ML_SERVER_SCRIPT="./express_canvas/start_ml_server.sh"
LOG_DIR="logs"
ML_LOG="$LOG_DIR/ml_server.log"
EXPRESS_LOG="$LOG_DIR/express_server.log"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DoodleParty with Content Detection${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: node not found${NC}"
    exit 1
fi

mkdir -p "$LOG_DIR"

export ML_MODEL_PATH
export ML_INPUT_SIZE
export ML_CONFIDENCE_THRESHOLD
export ML_SERVER_PORT="$ML_PORT"

# Kill any existing processes on these ports
echo -e "${YELLOW}Checking for existing processes...${NC}"
lsof -ti:$ML_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$EXPRESS_PORT | xargs kill -9 2>/dev/null || true
echo -e "${GREEN}✓ Ports cleared${NC}"
echo ""

# Start ML server in background via start_ml_server.sh
if [ ! -x "$ML_SERVER_SCRIPT" ]; then
    if [ -f "$ML_SERVER_SCRIPT" ]; then
        chmod +x "$ML_SERVER_SCRIPT"
    else
        echo -e "${RED}Error: ML server script not found at $ML_SERVER_SCRIPT${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}Starting ML server via $ML_SERVER_SCRIPT...${NC}"
bash "$ML_SERVER_SCRIPT" > "$ML_LOG" 2>&1 &
ML_PID=$!
echo -e "${GREEN}✓ ML server startup initiated (PID: $ML_PID)${NC}"
echo "  Log: $ML_LOG"
echo ""

# Wait for ML server to be ready
echo -e "${YELLOW}Waiting for ML server...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$ML_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ML server is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: ML server failed to start${NC}"
        kill $ML_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
echo ""

# Start Express server
echo -e "${YELLOW}Starting Express server on port $EXPRESS_PORT...${NC}"
cd express_canvas/server
node express-server.cjs > ../../$EXPRESS_LOG 2>&1 &
EXPRESS_PID=$!
cd ../..

echo -e "${GREEN}✓ Express server started (PID: $EXPRESS_PID)${NC}"
echo "  Log: $EXPRESS_LOG"
echo ""

# Wait for Express server to be ready
echo -e "${YELLOW}Waiting for Express server...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:$EXPRESS_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Express server is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: Express server failed to start${NC}"
        kill $ML_PID $EXPRESS_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done
echo ""

# Show status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  All servers running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "ML Server:      http://localhost:$ML_PORT"
echo -e "Web App:        http://localhost:$EXPRESS_PORT"
echo ""
echo -e "Process IDs:"
echo -e "  ML Server:    $ML_PID"
echo -e "  Express:      $EXPRESS_PID"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Create a trap to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $ML_PID $EXPRESS_PID 2>/dev/null || true
    echo -e "${GREEN}✓ Servers stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for user interrupt
wait
