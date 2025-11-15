#!/bin/bash

# DoodleHunter Web Interface Launcher
# Interactive menu to select model resolution before starting Flask server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Banner
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              🎨 DoodleHunter Web Interface 🎨              ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Function to check if model exists
check_model() {
    local model_path="$1"
    if [ -f "$model_path" ]; then
        local size=$(du -h "$model_path" | cut -f1)
        echo -e "${GREEN}✓${NC} Found ($size)"
        return 0
    else
        echo -e "${RED}✗${NC} Not found"
        return 1
    fi
}

# Display model availability
echo -e "${BLUE}Available Models:${NC}"
echo ""

echo -n "  1) 64x64 Resolution    "
if check_model "models/quickdraw_model_64x64.h5"; then
    MODEL_64_AVAILABLE=1
else
    MODEL_64_AVAILABLE=0
fi

echo -n "  2) 96x96 Resolution    "
if check_model "models/quickdraw_model_96x96.h5"; then
    MODEL_96_AVAILABLE=1
else
    MODEL_96_AVAILABLE=0
fi

echo -n "  3) 128x128 Resolution  "
if check_model "models/quickdraw_model.h5"; then
    MODEL_128_AVAILABLE=1
else
    MODEL_128_AVAILABLE=0
fi

echo ""

# Check TFLite models
echo -e "${BLUE}TFLite Models (optimized):${NC}"
echo ""

echo -n "  4) 64x64 TFLite Float32   "
if check_model "models/quickdraw_model_64x64.tflite"; then
    TFLITE_64_AVAILABLE=1
else
    TFLITE_64_AVAILABLE=0
fi

echo -n "  5) 96x96 TFLite Float32   "
if check_model "models/quickdraw_model_96x96.tflite"; then
    TFLITE_96_AVAILABLE=1
else
    TFLITE_96_AVAILABLE=0
fi

echo -n "  6) 128x128 TFLite Float32 "
if check_model "models/quickdraw_model.tflite"; then
    TFLITE_128_AVAILABLE=1
else
    TFLITE_128_AVAILABLE=0
fi

echo -n "  7) 64x64 TFLite INT8      "
if check_model "models/quickdraw_model_64x64_int8.tflite"; then
    TFLITE_64_INT8_AVAILABLE=1
else
    TFLITE_64_INT8_AVAILABLE=0
fi

echo -n "  8) 96x96 TFLite INT8      "
if check_model "models/quickdraw_model_96x96_int8.tflite"; then
    TFLITE_96_INT8_AVAILABLE=1
else
    TFLITE_96_INT8_AVAILABLE=0
fi

echo -n "  9) 128x128 TFLite INT8    "
if check_model "models/quickdraw_model_int8.tflite"; then
    TFLITE_128_INT8_AVAILABLE=1
else
    TFLITE_128_INT8_AVAILABLE=0
fi

echo ""
echo -e "  ${YELLOW}0) Exit${NC}"
echo ""

# Prompt for selection
while true; do
    echo -n -e "${CYAN}Select model to run (0-9): ${NC}"
    read -r choice
    
    case $choice in
        0)
            echo -e "${YELLOW}Exiting...${NC}"
            exit 0
            ;;
        1)
            if [ $MODEL_64_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=64
                MODEL_PATH="models/quickdraw_model_64x64.h5"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        2)
            if [ $MODEL_96_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=96
                MODEL_PATH="models/quickdraw_model_96x96.h5"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        3)
            if [ $MODEL_128_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=128
                MODEL_PATH="models/quickdraw_model.h5"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        4)
            if [ $TFLITE_64_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=64
                MODEL_PATH="models/quickdraw_model_64x64.tflite"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        5)
            if [ $TFLITE_96_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=96
                MODEL_PATH="models/quickdraw_model_96x96.tflite"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        6)
            if [ $TFLITE_128_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=128
                MODEL_PATH="models/quickdraw_model.tflite"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        7)
            if [ $TFLITE_64_INT8_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=64
                MODEL_PATH="models/quickdraw_model_64x64_int8.tflite"
                echo -e "${YELLOW}⚠️  Warning: 64x64 INT8 model has known accuracy issues (61% agreement)${NC}"
                echo -e "${YELLOW}   Recommended: Use option 4 (64x64 Float32) instead for better accuracy${NC}"
                echo -n -e "${CYAN}Continue anyway? (y/n): ${NC}"
                read -r confirm
                if [[ $confirm != "y" && $confirm != "Y" ]]; then
                    echo -e "${YELLOW}Selection cancelled. Please choose another option.${NC}"
                    continue
                fi
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        8)
            if [ $TFLITE_96_INT8_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=96
                MODEL_PATH="models/quickdraw_model_96x96_int8.tflite"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        9)
            if [ $TFLITE_128_INT8_AVAILABLE -eq 1 ]; then
                MODEL_SIZE=128
                MODEL_PATH="models/quickdraw_model_int8.tflite"
                break
            else
                echo -e "${RED}Model not available. Please select another option.${NC}"
            fi
            ;;
        *)
            echo -e "${RED}Invalid selection. Please enter a number between 0-9.${NC}"
            ;;
    esac
done

echo ""
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting DoodleHunter Web Interface${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo -e "  Model Resolution: ${CYAN}${MODEL_SIZE}x${MODEL_SIZE}${NC}"
echo -e "  Model Path:       ${CYAN}${MODEL_PATH}${NC}"
echo -e "  Port:             ${CYAN}5000${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
echo ""

# Set environment variables
export DOODLEHUNTER_MODEL_SIZE=$MODEL_SIZE
export DOODLEHUNTER_MODEL_PATH="$SCRIPT_DIR/$MODEL_PATH"

# Activate virtual environment and start Flask
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

echo -e "${BLUE}Starting Flask server...${NC}"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}    Access the web interface at:${NC}"
echo -e "${YELLOW}    • Local:   ${CYAN}http://127.0.0.1:5000${NC}"
echo -e "${YELLOW}    • Network: ${CYAN}http://$(hostname -I | awk '{print $1}'):5000${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${RED}Press Ctrl+C to stop the server${NC}"
echo ""

# Start Flask app
python3 src/web/app.py
