# DoodleParty Nix Setup

Complete NixOS module and Nix flake for DoodleParty.

## Files Created

- **`flake.nix`** - Nix flake with development shell, packages, and apps
- **`module.nix`** - NixOS/Home Manager module for systemd service
- **`default.nix`** - Backwards compatible build (non-flake)
- **`shell.nix`** - Development shell (non-flake)
- **`nixos-example.nix`** - Example NixOS configurations
- **`.envrc`** - direnv integration for automatic environment loading

## Quick Start

### Option 1: With Flakes (Recommended)

```bash
# Enter development environment
nix develop

# Run the application
nix run

# Build the package
nix build
```

### Option 2: Without Flakes

```bash
# Enter development environment
nix-shell

# Build the package
nix-build
```

### Option 3: With direnv (Auto-load on cd)

```bash
# Install direnv
nix-env -iA nixpkgs.direnv

# Enable in your shell (add to ~/.bashrc or ~/.zshrc)
eval "$(direnv hook bash)"  # or zsh

# Allow direnv in this directory
cd /home/diatom/Documents/DoodleParty
direnv allow

# Now the environment loads automatically when you cd here
```

## NixOS System Service

### Using Flakes

Add to your `flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    doodleparty.url = "path:/home/diatom/Documents/DoodleParty";
  };

  outputs = { self, nixpkgs, doodleparty }: {
    nixosConfigurations.yourhostname = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        doodleparty.nixosModules.doodleparty
      ];
    };
  };
}
```

Then in `configuration.nix`:

```nix
{
  services.doodleparty = {
    enable = true;
    port = 5000;
    openFirewall = true;
  };
}
```

### Without Flakes

In your `/etc/nixos/configuration.nix`:

```nix
{ config, pkgs, ... }:

{
  imports = [
    /home/diatom/Documents/DoodleParty/module.nix
  ];

  services.doodleparty = {
    enable = true;
    port = 5000;
    openFirewall = true;
  };
}
```

Then rebuild:

```bash
sudo nixos-rebuild switch
```

## Configuration Options

### Basic Options

```nix
services.doodleparty = {
  enable = true;              # Enable the service
  host = "0.0.0.0";          # Bind address
  port = 5000;                # Port number
  openFirewall = true;        # Open firewall port
};
```

### Advanced Options

```nix
services.doodleparty = {
  enable = true;
  
  # Performance
  workers = 4;                      # Gunicorn workers
  maxConcurrentUsers = 100;         # Max concurrent users
  
  # AI/ML
  enableAI = true;                  # Content moderation
  enableLLM = false;                # LLM integration
  moderationThreshold = 0.85;       # 0.0 - 1.0
  
  # Secrets (secure way)
  llmApiKeyFile = "/run/secrets/llm-key";
  
  # Logging
  logLevel = "INFO";                # DEBUG, INFO, WARNING, ERROR, CRITICAL
  enableMetrics = true;             # Prometheus metrics
  
  # Directories
  dataDir = "/var/lib/doodleparty";
  modelsDir = "/var/lib/doodleparty/models";
  
  # Custom config
  extraConfig = {
    CANVAS_WIDTH = 1920;
    CANVAS_HEIGHT = 1080;
    SAVE_INTERVAL = 60;
  };
};
```

### Raspberry Pi 4 Configuration

```nix
services.doodleparty = {
  enable = true;
  host = "0.0.0.0";
  port = 5000;
  openFirewall = true;
  
  # Optimize for RPi4
  workers = 2;
  maxConcurrentUsers = 100;
  enableLLM = false;  # Save resources
  
  extraConfig = {
    MODEL_TYPE = "tflite";
    USE_INT8_QUANTIZATION = "true";
    CANVAS_WIDTH = 1280;
    CANVAS_HEIGHT = 720;
  };
};
```

### High-Performance Server

```nix
services.doodleparty = {
  enable = true;
  workers = 16;
  maxConcurrentUsers = 1000;
  enableAI = true;
  enableLLM = true;
  
  extraConfig = {
    CANVAS_WIDTH = 3840;
    CANVAS_HEIGHT = 2160;
    WORKER_TIMEOUT = 300;
    MAX_BATCH_SIZE = 64;
  };
};
```

## Development Workflow

### 1. Enter Development Shell

```bash
nix develop
# or: nix-shell
# or: just cd here if using direnv
```

### 2. Install Dependencies

```bash
npm install
# Python deps are already in Nix environment
```

### 3. Start Development Server

```bash
npm run dev
```

### 4. Run Tests

```bash
# Frontend tests
npm run test

# Python tests
python -m pytest src-py/

# Type checking
npm run type-check

# Linting
npm run lint
python -m flake8 src-py/
```

### 5. Build for Production

```bash
npm run build
nix build
```

## Service Management

```bash
# Check status
systemctl status doodleparty

# View logs
journalctl -u doodleparty -f

# Restart
systemctl restart doodleparty

# Enable on boot
systemctl enable doodleparty
```

## With Nginx Reverse Proxy

The module automatically configures Nginx if enabled:

```nix
services.nginx.enable = true;

services.nginx.virtualHosts."doodleparty.example.com" = {
  serverName = "doodleparty.example.com";
  enableACME = true;
  forceSSL = true;
};

security.acme = {
  acceptTerms = true;
  defaults.email = "admin@example.com";
};
```

## Secrets Management

### Option 1: agenix

```nix
{ config, ... }:

{
  age.secrets.doodleparty-llm-key = {
    file = ./secrets/llm-key.age;
    owner = "doodleparty";
  };
  
  services.doodleparty = {
    enable = true;
    enableLLM = true;
    llmApiKeyFile = config.age.secrets.doodleparty-llm-key.path;
  };
}
```

### Option 2: sops-nix

```nix
{ config, ... }:

{
  sops.secrets.doodleparty-llm-key = {
    owner = "doodleparty";
  };
  
  services.doodleparty = {
    enable = true;
    enableLLM = true;
    llmApiKeyFile = config.sops.secrets.doodleparty-llm-key.path;
  };
}
```

### Option 3: Plain file (development only)

```bash
# Create secret file
echo "sk-your-api-key-here" > /run/secrets/llm-key
chmod 600 /run/secrets/llm-key
chown doodleparty:doodleparty /run/secrets/llm-key
```

## Building for Different Platforms

### For x86_64 Linux (default)

```bash
nix build
```

### For Raspberry Pi 4 (ARM)

```bash
nix build --system aarch64-linux
```

### Cross-compilation

```nix
# In your configuration
nixpkgs.crossSystem = {
  config = "aarch64-unknown-linux-gnu";
};
```

## Troubleshooting

### Flake issues

```bash
# Update flake lock
nix flake update

# Check flake
nix flake check

# Show flake info
nix flake show
```

### Service not starting

```bash
# Check logs
journalctl -u doodleparty -n 50

# Check service file
systemctl cat doodleparty

# Test manually
sudo -u doodleparty /nix/store/.../bin/doodleparty
```

### Build failures

```bash
# Clear cache
nix-collect-garbage

# Rebuild with verbose output
nix build --verbose --show-trace

# Check build log
nix log
```

### Missing dependencies

If a Python package is missing, add it to `pythonEnv` in both `flake.nix` and `module.nix`:

```nix
pythonEnv = pkgs.python3.withPackages (ps: with ps; [
  # existing packages...
  your-new-package
]);
```

## Integration with Existing Tools

### With VS Code

Install "Nix IDE" extension, then:

```json
// .vscode/settings.json
{
  "nix.enableLanguageServer": true,
  "nix.serverPath": "nil",
  "python.defaultInterpreterPath": "${workspaceFolder}/.direnv/python/bin/python"
}
```

### With Home Manager

```nix
{ config, pkgs, ... }:

{
  home.packages = [
    (import /home/diatom/Documents/DoodleParty {})
  ];
}
```

### With Docker/Podman

```nix
# Build Docker image
nix build .#dockerImage

# Load image
docker load < result
```

## Package Structure

```
DoodleParty/
├── flake.nix          # Main flake definition
├── module.nix         # NixOS module
├── default.nix        # Non-flake build
├── shell.nix          # Non-flake dev shell
├── nixos-example.nix  # Example configs
└── .envrc             # direnv config
```

## Resources

- **[Full Documentation](.documentation/nix-usage.md)** - Comprehensive guide
- **[Installation Guide](.documentation/installation.md)** - Setup instructions
- **[Architecture](.documentation/architecture.md)** - System design
- **[API Reference](.documentation/api.md)** - API documentation
- **[NixOS Manual](https://nixos.org/manual/nixos/stable/)** - NixOS docs
- **[Nix Manual](https://nixos.org/manual/nix/stable/)** - Nix package manager

## Contributing

When modifying Nix files:

1. Test in dev shell: `nix develop`
2. Build package: `nix build`
3. Test module: `nixos-rebuild test` (on NixOS)
4. Format: `nix fmt` or `nixpkgs-fmt *.nix`
5. Check: `nix flake check`

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE)
