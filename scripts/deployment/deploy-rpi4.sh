#!/bin/bash
# Deploy DoodleParty to Raspberry Pi 4

set -e

echo "DoodleParty RPi4 Deployment"
echo "============================"

# Configuration
RPI_HOST=${RPI_HOST:-"doodleparty-pi.local"}
RPI_USER=${RPI_USER:-"pi"}
DEPLOY_DIR="/home/pi/doodleparty"

echo "Target: $RPI_USER@$RPI_HOST"
echo "Deploy directory: $DEPLOY_DIR"

# Copy files to RPi
echo "Copying files..."
rsync -avz --exclude '.git' --exclude 'node_modules' --exclude '.venv' . "$RPI_USER@$RPI_HOST:$DEPLOY_DIR"

# Install dependencies
echo "Installing dependencies..."
ssh "$RPI_USER@$RPI_HOST" "cd $DEPLOY_DIR && pip install -r requirements-rpi4.txt"

echo "Deployment complete!"
