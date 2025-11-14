# DoodleParty Installation Guide

**Purpose:** Complete installation and setup instructions for development and deployment.

**Status: Updated to match actual implementation**

## Table of Contents

### Preparation
- [Prerequisites](#prerequisites)
  - [Development Environment](#development-environment)
  - [Raspberry Pi 4 Deployment](#raspberry-pi-4-deployment)

### Installation
- [Installation Steps](#installation-steps)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Development Environment Setup](#2-development-environment-setup)
    - [With Nix (Recommended)](#with-nix-recommended)
    - [Without Nix](#without-nix)
  - [3. Download ML Model](#3-download-ml-model)
  - [4. Start Development Server](#4-start-development-server)
  - [5. Start ML Inference Service](#5-start-ml-inference-service)

### Configuration
- [Environment Variables](#environment-variables)
- [Data Directory Structure](#data-directory-structure)

### Deployment Options
- [Raspberry Pi 4 Deployment](#raspberry-pi-4-deployment-1)
  - [1. Prepare Raspberry Pi OS](#1-prepare-raspberry-pi-os)
  - [2. Install System Dependencies](#2-install-system-dependencies)
  - [3. Clone and Setup DoodleParty](#3-clone-and-setup-doodleparty)
  - [4. Download ML Model](#4-download-ml-model-1)
  - [5. Configure System Performance](#5-configure-system-performance)
  - [6. Create Systemd Services](#6-create-systemd-services)
    - [Node.js Service](#nodejs-service)
    - [ML Service](#ml-service)
    - [Enable services](#enable-services)
  - [7. Verify Installation](#7-verify-installation)

- [Cloud Deployment (DigitalOcean)](#cloud-deployment-digitalocean)
  - [1. Create Kubernetes Cluster](#1-create-kubernetes-cluster)
  - [2. Deploy Application](#2-deploy-application)
  - [3. Setup Database](#3-setup-database)
  - [4. Setup Redis Cache](#4-setup-redis-cache)


### Next Steps
- [Further Documentation](#further-documentation)

## Prerequisites

### Development Environment

- Node.js 18+ and npm
- Python 3.9+ (for ML pipeline)
- Git
- 4GB+ RAM
- 2GB+ disk space
- Internet connection

### Raspberry Pi 4 Deployment

- Raspberry Pi 4 Model B (4GB or 8GB RAM recommended)
- 32GB microSD card (UHS-I A1 or better)
- 5V 3A USB-C power supply with surge protection
- Active cooling (heatsink + 30mm fan)
- Ventilated case
- Ethernet cable or WiFi adapter

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/doodleparty.git
cd doodleparty
```

### 2. Development Environment Setup

#### With Nix (Recommended)

```bash
nix develop
```

This provides:
- Node.js 18 with all required packages
- Python 3.11 with TensorFlow and dependencies
- Development tools (eslint, prettier, pytest)

#### Without Nix

**Install Node.js dependencies:**
```bash
npm install
```

**Install Python dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download ML Model

The ML model is required for content moderation. For detailed information about the model architecture and performance, see the [ML Pipeline documentation](ml-pipeline.md).

```bash
python scripts/data_processing/download_quickdraw_npy.py
python scripts/train.py --epochs 50 --batch-size 32
```

Or download pre-trained model:
```bash
wget https://releases.doodleparty.io/models/quickdraw_model_int8.tflite
mkdir -p models
mv quickdraw_model_int8.tflite models/
```

### 4. Start Development Server

```bash
npm run dev
```

Access at `http://localhost:3000`

### 5. Start ML Inference Service

In a separate terminal:
```bash
python src/web/app.py
```

ML service runs on `http://localhost:5001`

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Node.js Server
NODE_ENV=development
PORT=3000
ML_SERVICE_URL=http://localhost:5001

# ML Model
MODEL_PATH=models/quickdraw_model_int8.tflite
IMAGE_SIZE=28
THRESHOLD=0.5

# Database (if using cloud)
DATABASE_URL=postgresql://user:password@localhost/doodleparty
REDIS_URL=redis://localhost:6379

# DigitalOcean AI (optional)
DIGITALOCEAN_API_KEY=your_api_key_here
```

### Data Directory Structure

```
doodleparty/
├── data/
│   ├── raw/
│   │   ├── penis.npy
│   │   ├── circle.npy
│   │   └── ...
│   └── processed/
│       └── class_mapping.pkl
├── models/
│   ├── quickdraw_model.h5
│   ├── quickdraw_model.tflite
│   └── quickdraw_model_int8.tflite
└── logs/
```

### Raspberry Pi 4 Deployment

This guide covers a standard RPi4 deployment. For NixOS-specific instructions, see the [Nix Usage Guide](nix-usage.md#raspberry-pi-4-with-nixos). For performance optimization details, see the [ML Pipeline documentation](ml-pipeline.md#raspberry-pi-4-optimization).

### 1. Prepare Raspberry Pi OS

**Install Raspberry Pi OS Lite (64-bit):**
- Download from https://www.raspberrypi.com/software/
- Flash to microSD card using Raspberry Pi Imager
- Enable SSH during setup

**Connect to RPi4:**
```bash
ssh pi@192.168.1.10
```

### 2. Install System Dependencies

```bash
sudo apt update
sudo apt upgrade
sudo apt install -y python3-pip python3-venv nodejs npm
```

### 3. Clone and Setup DoodleParty

```bash
cd /home/pi
git clone https://github.com/yourusername/doodleparty.git
cd doodleparty

# Install Node.js dependencies
npm install --production

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements-rpi4.txt
```

### 4. Download ML Model

```bash
mkdir -p models
wget https://releases.doodleparty.io/models/quickdraw_model_int8.tflite
mv quickdraw_model_int8.tflite models/
```

### 5. Configure System Performance

**Set CPU governor to performance mode (CRITICAL):**
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Make persistent
sudo nano /etc/rc.local
# Add before 'exit 0':
# echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**Free up memory by disabling unnecessary services (saves ~25MB):**
```bash
# Disable Bluetooth (~10MB)
sudo systemctl disable bluetooth
sudo systemctl stop bluetooth

# Disable Avahi mDNS (~5MB)
sudo systemctl disable avahi-daemon
sudo systemctl stop avahi-daemon

# Disable HDMI if no display (~10MB)
/opt/vc/bin/tvservice -o

# Verify freed memory
free -h
```

**Disable swap for stability (optional but recommended):**
```bash
# Swap on microSD is extremely slow; better to crash cleanly
sudo dphys-swapfile swapoff
sudo dphys-swapfile uninstall
sudo update-rc.d dphys-swapfile remove

# Verify
free -h
```

**Enable active cooling monitoring:**
```bash
# Install vcgencmd for temperature monitoring
sudo apt install -y libraspberrypi-bin

# Monitor temperature
vcgencmd measure_temp
```

**Memory Budget After Optimization (4GB RAM):**
- OS + services: ~500-800MB
- Node.js server: 256-512MB (with `--max-old-space-size=512`)
- ML service: 300-500MB (TFLite INT8)
- Available for users: ~1.5-2GB (100 concurrent users)

### 6. Create Systemd Services

**Node.js Service (`/etc/systemd/system/doodleparty-web.service`):**
```ini
[Unit]
Description=DoodleParty Web Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/doodleparty
Environment="NODE_OPTIONS=--max-old-space-size=512 --expose-gc"
ExecStart=/usr/bin/node src/server.js
Restart=on-failure
RestartSec=10
Nice=-10
CPUAffinity=0-3

[Install]
WantedBy=multi-user.target
```

**ML Service (`/etc/systemd/system/doodleparty-ml.service`):**
```ini
[Unit]
Description=DoodleParty ML Inference Service
After=network.target doodleparty-web.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/doodleparty
Environment="PATH=/home/pi/doodleparty/venv/bin"
ExecStart=/home/pi/doodleparty/venv/bin/python src/web/app.py
Restart=on-failure
RestartSec=10
Nice=-5
CPUAffinity=0-3

[Install]
WantedBy=multi-user.target
```

**Configuration Notes:**
- `NODE_OPTIONS`: Limit Node.js heap to 512MB, expose garbage collection
- `Nice=-10`: Higher priority for web server (lower number = higher priority)
- `Nice=-5`: Slightly lower priority for ML service (still high)
- `CPUAffinity=0-3`: Use all 4 RPi4 cores
- `After=network.target`: Ensure network is ready before starting

**Enable services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable doodleparty-web
sudo systemctl enable doodleparty-ml
sudo systemctl start doodleparty-web
sudo systemctl start doodleparty-ml
```

### 7. Verify Installation

```bash
# Check services status
sudo systemctl status doodleparty-web
sudo systemctl status doodleparty-ml

# Check temperature
vcgencmd measure_temp

# Access web interface
# Open browser and go to http://192.168.1.10:3000
```

## Cloud Deployment (DigitalOcean)

### 1. Create Kubernetes Cluster

```bash
doctl kubernetes cluster create doodleparty \
  --region nyc3 \
  --node-pool name=web-pool count=2 size=s-2vcpu-4gb \
  --node-pool name=gpu-pool count=1 size=gpu-nvidia-tesla-t4 gpu-count=1
```

### 2. Deploy Application

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment-web.yaml
kubectl apply -f k8s/deployment-ml.yaml
kubectl apply -f k8s/service.yaml
```

### 3. Setup Database

```bash
doctl databases create doodleparty-db \
  --engine pg \
  --region nyc3 \
  --num-nodes 1
```

### 4. Setup Redis Cache

```bash
doctl databases create doodleparty-redis \
  --engine redis \
  --region nyc3 \
  --num-nodes 1
```


## Next Steps

1. Review [Architecture](architecture.md) for system design
2. Check [API Reference](api.md) for WebSocket events
3. See [Development Roadmap](roadmap.md) for planned features
4. Understand the [ML Pipeline](ml-pipeline.md) for content moderation
5. Review [Testing Strategy](testing.md) for development guidelines
6. For Nix-specific deployment, see [Nix Usage Guide](nix-usage.md)
7. Check [Project Structure](structure.md) to understand code organization

*Installation guide for DoodleParty v1.0*
