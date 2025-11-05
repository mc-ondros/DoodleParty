# DoodleHunter Nix Usage Guide

**Purpose:** Instructions for using DoodleHunter with Nix and NixOS.

## Quick Start with Nix Flakes

### Development Shell

Enter a development environment with all dependencies:

```bash
nix develop
```

This provides:
- Python 3.11 with all required packages
- TensorFlow, NumPy, Pandas, Matplotlib
- Flask for web interface
- Development tools (pytest, black, flake8, mypy)

### Running Applications

**Train a model:**
```bash
nix run .#train -- --epochs 50 --batch-size 32
```

**Evaluate a model:**
```bash
nix run .#evaluate -- --model models/quickdraw_classifier.h5
```

**Start web interface:**
```bash
nix run .#web
```

### Building the Package

Build DoodleHunter as a Python package:

```bash
nix build
```

The result will be in `./result/`.

## NixOS Module

### Installation

Add to your NixOS configuration:

```nix
# configuration.nix
{ config, pkgs, ... }:

{
  imports = [
    /path/to/doodlehunter/module.nix
  ];

  services.doodlehunter = {
    enable = true;
    port = 5000;
    host = "0.0.0.0";
    modelPath = "/var/lib/doodlehunter/models/quickdraw_classifier.h5";
    openFirewall = true;
  };
}
```

### Configuration Options

**`services.doodlehunter.enable`**
- Type: boolean
- Default: `false`
- Enable the DoodleHunter web interface service

**`services.doodlehunter.port`**
- Type: port
- Default: `5000`
- Port for the Flask web interface

**`services.doodlehunter.host`**
- Type: string
- Default: `"127.0.0.1"`
- Host address to bind to (use `"0.0.0.0"` for external access)

**`services.doodlehunter.modelPath`**
- Type: path
- Required
- Path to the trained model file

**`services.doodlehunter.dataDir`**
- Type: path
- Default: `"/var/lib/doodlehunter"`
- Directory for DoodleHunter data

**`services.doodlehunter.user`**
- Type: string
- Default: `"doodlehunter"`
- User account under which DoodleHunter runs

**`services.doodlehunter.group`**
- Type: string
- Default: `"doodlehunter"`
- Group under which DoodleHunter runs

**`services.doodlehunter.openFirewall`**
- Type: boolean
- Default: `false`
- Whether to open the firewall for the web interface

### Example Configurations

**Basic local development:**
```nix
services.doodlehunter = {
  enable = true;
  modelPath = "/home/user/models/quickdraw_classifier.h5";
};
```

**Production deployment:**
```nix
services.doodlehunter = {
  enable = true;
  port = 8080;
  host = "0.0.0.0";
  modelPath = "/var/lib/doodlehunter/models/quickdraw_classifier.h5";
  openFirewall = true;
};
```

### Service Management

**Start the service:**
```bash
sudo systemctl start doodlehunter
```

**Check status:**
```bash
sudo systemctl status doodlehunter
```

**View logs:**
```bash
sudo journalctl -u doodlehunter -f
```

**Restart the service:**
```bash
sudo systemctl restart doodlehunter
```

## Development Workflow

### 1. Enter Development Shell

```bash
cd /path/to/DoodleHunter
nix develop
```

### 2. Train Model

```bash
python scripts/train.py --epochs 50
```

### 3. Test Web Interface

```bash
python src/web/app.py
```

### 4. Run Tests

```bash
pytest tests/
```

### 5. Format Code

```bash
black src/ scripts/ tests/
```

## Flake Structure

The flake provides:

**Packages:**
- `default` - DoodleHunter Python package

**Development Shells:**
- `default` - Full development environment

**Apps:**
- `train` - Training script
- `evaluate` - Evaluation script
- `web` - Web interface

## Troubleshooting

### TensorFlow Not Found

If TensorFlow is not available in the Nix shell:

```bash
# Check Python packages
python -c "import tensorflow; print(tensorflow.__version__)"

# If missing, ensure flake.nix includes tensorflow in propagatedBuildInputs
```

### Model Path Issues

Ensure the model path is accessible:

```bash
# Check file exists
ls -l /var/lib/doodlehunter/models/quickdraw_classifier.h5

# Check permissions
sudo chown doodlehunter:doodlehunter /var/lib/doodlehunter/models/
```

### Port Already in Use

Change the port in your configuration:

```nix
services.doodlehunter.port = 8080;
```

## Related Documentation

- [Installation](installation.md) - Standard installation
- [Architecture](architecture.md) - System design
- [API Reference](api.md) - API documentation

*Nix usage guide for DoodleHunter v1.0*
