# Example NixOS Configuration for DoodleParty
# Copy this to your /etc/nixos/configuration.nix or import it as a module

{ config, pkgs, lib, ... }:

{
  # Import the DoodleParty module
  imports = [
    # If using flakes, this is not needed
    # Otherwise, specify the path to module.nix
    # /path/to/DoodleParty/module.nix
  ];

  # Basic configuration - suitable for testing/development
  services.doodleparty = {
    enable = true;
    port = 5000;
    openFirewall = true;
  };

  # Advanced configuration example - commented out
  # Uncomment and modify as needed for production use
  
  # services.doodleparty = {
  #   enable = true;
  #   
  #   # Network settings
  #   host = "0.0.0.0";
  #   port = 5000;
  #   openFirewall = true;
  #   
  #   # User/group
  #   user = "doodleparty";
  #   group = "doodleparty";
  #   
  #   # Directories
  #   dataDir = "/var/lib/doodleparty";
  #   modelsDir = "/var/lib/doodleparty/models";
  #   
  #   # Performance tuning
  #   workers = 4;  # Adjust based on CPU cores
  #   maxConcurrentUsers = 100;  # Increase for larger events
  #   
  #   # AI/ML settings
  #   enableAI = true;
  #   moderationThreshold = 0.85;  # Lower = more strict (0.0-1.0)
  #   
  #   # LLM integration (optional - requires API key)
  #   enableLLM = false;
  #   # Store API key securely in a separate file
  #   # llmApiKeyFile = "/run/secrets/doodleparty-llm-key";
  #   
  #   # Logging and monitoring
  #   logLevel = "INFO";  # DEBUG | INFO | WARNING | ERROR | CRITICAL
  #   enableMetrics = true;  # Prometheus metrics endpoint
  #   
  #   # Custom configuration
  #   extraConfig = {
  #     CANVAS_WIDTH = 1920;
  #     CANVAS_HEIGHT = 1080;
  #     SAVE_INTERVAL = 60;  # seconds
  #   };
  # };

  # Optional: Nginx reverse proxy with SSL
  # services.nginx = {
  #   enable = true;
  #   
  #   virtualHosts."doodleparty.example.com" = {
  #     serverName = "doodleparty.example.com";
  #     
  #     # SSL with Let's Encrypt
  #     enableACME = true;
  #     forceSSL = true;
  #     
  #     # The DoodleParty module automatically sets up the proxy
  #     # when nginx is enabled
  #   };
  # };
  # 
  # security.acme = {
  #   acceptTerms = true;
  #   defaults.email = "admin@example.com";
  # };

  # Optional: Prometheus monitoring
  # services.prometheus = {
  #   enable = true;
  #   scrapeConfigs = [
  #     {
  #       job_name = "doodleparty";
  #       static_configs = [{
  #         targets = [ "localhost:5000" ];
  #       }];
  #     }
  #   ];
  # };

  # Optional: Fail2ban for brute-force protection
  # services.fail2ban = {
  #   enable = true;
  #   jails.doodleparty = ''
  #     enabled = true
  #     filter = doodleparty
  #     logpath = /var/log/doodleparty/access.log
  #     maxretry = 5
  #     bantime = 3600
  #   '';
  # };
}

# Raspberry Pi 4 specific configuration
# {
#   services.doodleparty = {
#     enable = true;
#     host = "0.0.0.0";
#     port = 5000;
#     openFirewall = true;
#     
#     # Optimize for RPi4 resources
#     workers = 2;
#     maxConcurrentUsers = 100;
#     
#     # Use INT8 quantized models for better performance
#     extraConfig = {
#       MODEL_TYPE = "tflite";
#       USE_INT8_QUANTIZATION = "true";
#       
#       # Reduce memory usage
#       CANVAS_WIDTH = 1280;
#       CANVAS_HEIGHT = 720;
#       JPEG_QUALITY = 75;
#       
#       # Optimize for embedded device
#       WORKER_TIMEOUT = 60;
#       MAX_BATCH_SIZE = 16;
#     };
#     
#     # Disable LLM for offline events
#     enableLLM = false;
#   };
#   
#   # Performance tuning for RPi4
#   boot.kernelParams = [ "cma=256M" ];
#   
#   # Disable unnecessary services to save resources
#   services.avahi.enable = false;
#   services.printing.enable = false;
# }

# High-concurrency event configuration (powerful server)
# {
#   services.doodleparty = {
#     enable = true;
#     host = "0.0.0.0";
#     port = 5000;
#     openFirewall = true;
#     
#     # Scale up for large events
#     workers = 16;
#     maxConcurrentUsers = 1000;
#     
#     # Full features enabled
#     enableAI = true;
#     enableLLM = true;
#     llmApiKeyFile = "/run/secrets/doodleparty-llm-key";
#     moderationThreshold = 0.80;
#     
#     extraConfig = {
#       # High quality canvas
#       CANVAS_WIDTH = 3840;
#       CANVAS_HEIGHT = 2160;
#       JPEG_QUALITY = 90;
#       
#       # Performance optimizations
#       WORKER_TIMEOUT = 300;
#       MAX_BATCH_SIZE = 64;
#       CANVAS_UPDATE_INTERVAL = 50;
#       
#       # Enable all game modes
#       ENABLE_SPEED_SKETCH = "true";
#       ENABLE_GUESS_DOODLE = "true";
#       ENABLE_BATTLE_ROYALE = "true";
#       ENABLE_STORY_CANVAS = "true";
#     };
#   };
#   
#   # Resource limits for the service
#   systemd.services.doodleparty.serviceConfig = {
#     MemoryMax = "8G";
#     CPUQuota = "800%";  # 8 cores max
#   };
# }
