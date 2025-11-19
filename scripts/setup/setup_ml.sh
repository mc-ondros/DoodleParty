#!/bin/bash
# Setup script for DoodleParty ML system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================================================"
echo "🔧 DOODLEPARTY ML SYSTEM SETUP"
echo "========================================================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment already exists at venv/"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists at .venv/"
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
fi

echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ Pip upgraded"
echo ""

# Install ML server dependencies
echo "Installing ML server dependencies..."
pip install -r requirements-ml-server.txt

echo ""
echo "========================================================================"
echo "✅ SETUP COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Start all services:"
echo "     ./start_doodleparty.sh"
echo ""
echo "  2. Open browser:"
echo "     http://localhost:3000/doodleparty"
echo ""
echo "  3. Draw something and click '🔍 Detect' to test ML detection"
echo ""
