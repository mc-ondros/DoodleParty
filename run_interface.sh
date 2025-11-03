#!/bin/bash

# DoodleHunter ML Testing Interface Startup Script

echo "🎨 Starting DoodleHunter ML Testing Interface..."
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run this from the ML project root directory."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

echo "Checking dependencies..."

# Install Flask if not already installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask and dependencies..."
    pip install -r app/requirements.txt
fi

echo ""
echo "✓ Environment ready!"
echo ""
echo "Starting Flask server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌐 Interface: http://localhost:5000"
echo "📊 Health Check: http://localhost:5000/api/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run the Flask app
cd app
python3 app.py
