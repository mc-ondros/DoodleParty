{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.doodlehunter;
  python = pkgs.python311;
  pythonPackages = python.pkgs;
  
  doodlehunter = pythonPackages.buildPythonPackage {
    pname = "doodlehunter";
    version = "1.0.0";
    src = ./.;
    format = "pyproject";

    nativeBuildInputs = with pythonPackages; [
      setuptools
      wheel
    ];

    propagatedBuildInputs = with pythonPackages; [
      tensorflow
      keras
      numpy
      pandas
      matplotlib
      scikit-learn
      pillow
      flask
      flask-cors
      gunicorn
      tqdm
      requests
      seaborn
      opencv4
    ];

    doCheck = false;
  };

in {
  options.services.doodlehunter = {
    enable = mkEnableOption "DoodleHunter web interface";

    port = mkOption {
      type = types.port;
      default = 5000;
      description = "Port for the Flask web interface";
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Host address to bind to";
    };

    debug = mkOption {
      type = types.bool;
      default = false;
      description = "Enable Flask debug mode (not recommended for production)";
    };

    modelsDir = mkOption {
      type = types.path;
      description = "Directory containing trained model files (.tflite, .h5, .keras)";
      example = "/var/lib/doodlehunter/models";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/doodlehunter";
      description = "Base directory for DoodleHunter data and logs";
    };

    processedDataDir = mkOption {
      type = types.path;
      default = "/var/lib/doodlehunter/data/processed";
      description = "Directory for processed data and class mappings";
    };

    logDir = mkOption {
      type = types.path;
      default = "/var/lib/doodlehunter/logs";
      description = "Directory for application logs";
    };

    logLevel = mkOption {
      type = types.enum [ "DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL" ];
      default = "INFO";
      description = "Logging level for the application";
    };

    threshold = mkOption {
      type = types.float;
      default = 0.5;
      description = "Classification threshold (0.0-1.0) for binary classification";
      example = 0.5;
    };

    tfliteThreads = mkOption {
      type = types.int;
      default = 4;
      description = "Number of threads for TensorFlow Lite inference (set to CPU core count)";
    };

    enableRegionDetection = mkOption {
      type = types.bool;
      default = true;
      description = "Enable region-based detection mode (analyzes multiple patches)";
    };

    enableTileDetection = mkOption {
      type = types.bool;
      default = true;
      description = "Enable tile-based detection mode (grid partitioning)";
    };

    tileGridSize = mkOption {
      type = types.int;
      default = 8;
      description = "Grid size for tile-based detection (e.g., 8 = 8x8 = 64 tiles)";
    };

    canvasSize = mkOption {
      type = types.int;
      default = 512;
      description = "Canvas size in pixels (width and height)";
    };

    imageSize = mkOption {
      type = types.int;
      default = 128;
      description = "Model input image size in pixels";
    };

    user = mkOption {
      type = types.str;
      default = "doodlehunter";
      description = "User account under which DoodleHunter runs";
    };

    group = mkOption {
      type = types.str;
      default = "doodlehunter";
      description = "Group under which DoodleHunter runs";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Whether to open the firewall for the web interface";
    };

    workers = mkOption {
      type = types.int;
      default = 1;
      description = "Number of worker processes for production deployment (requires gunicorn)";
    };

    useGunicorn = mkOption {
      type = types.bool;
      default = false;
      description = "Use Gunicorn WSGI server instead of Flask development server";
    };
  };

  config = mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
      description = "DoodleHunter service user";
    };

    users.groups.${cfg.group} = {};

    # Ensure directories exist with correct permissions
    systemd.tmpfiles.rules = [
      "d '${cfg.dataDir}' 0755 ${cfg.user} ${cfg.group} - -"
      "d '${cfg.logDir}' 0755 ${cfg.user} ${cfg.group} - -"
      "d '${cfg.processedDataDir}' 0755 ${cfg.user} ${cfg.group} - -"
      "d '${cfg.modelsDir}' 0755 ${cfg.user} ${cfg.group} - -"
    ];

    systemd.services.doodlehunter = {
      description = "DoodleHunter Web Interface";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      preStart = ''
        # Ensure all required directories exist with correct permissions
        mkdir -p ${cfg.dataDir}
        mkdir -p ${cfg.logDir}
        mkdir -p ${cfg.processedDataDir}
        mkdir -p ${cfg.modelsDir}
        
        # Set ownership
        chown -R ${cfg.user}:${cfg.group} ${cfg.dataDir}
        chown -R ${cfg.user}:${cfg.group} ${cfg.logDir}
        chown -R ${cfg.user}:${cfg.group} ${cfg.processedDataDir}
        
        # Set permissions
        chmod 755 ${cfg.dataDir}
        chmod 755 ${cfg.logDir}
        chmod 755 ${cfg.processedDataDir}
        chmod 755 ${cfg.modelsDir}
      '';

      environment = {
        # Flask configuration
        FLASK_PORT = toString cfg.port;
        FLASK_HOST = cfg.host;
        FLASK_DEBUG = if cfg.debug then "1" else "0";
        
        # Model and data paths
        MODELS_DIR = cfg.modelsDir;
        DATA_DIR = cfg.dataDir;
        PROCESSED_DATA_DIR = cfg.processedDataDir;
        LOG_DIR = cfg.logDir;
        
        # Model configuration
        THRESHOLD = toString cfg.threshold;
        TFLITE_THREADS = toString cfg.tfliteThreads;
        IMAGE_SIZE = toString cfg.imageSize;
        
        # Detection modes
        ENABLE_REGION_DETECTION = if cfg.enableRegionDetection then "1" else "0";
        ENABLE_TILE_DETECTION = if cfg.enableTileDetection then "1" else "0";
        TILE_GRID_SIZE = toString cfg.tileGridSize;
        CANVAS_SIZE = toString cfg.canvasSize;
        
        # Logging
        LOG_LEVEL = cfg.logLevel;
        
        # Python path
        PYTHONPATH = "${doodlehunter}/${python.sitePackages}";
      };

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        
        # Use gunicorn for production or Flask dev server
        ExecStart = if cfg.useGunicorn
          then "${pythonPackages.gunicorn}/bin/gunicorn -w ${toString cfg.workers} -b ${cfg.host}:${toString cfg.port} --timeout 120 src.web.app:app"
          else "${python}/bin/python ${doodlehunter}/${python.sitePackages}/src/web/app.py";
        
        Restart = "on-failure";
        RestartSec = "5s";
        
        # Timeout configuration for ML inference
        TimeoutStartSec = "60s";
        TimeoutStopSec = "30s";

        # Security hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir cfg.logDir cfg.processedDataDir cfg.modelsDir ];
        
        # Resource limits (important for Raspberry Pi 4)
        MemoryMax = "2G";
        CPUQuota = "400%";  # Allow use of all 4 cores
        
        # Additional hardening
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictNamespaces = true;
        LockPersonality = true;
        RestrictRealtime = true;
        PrivateDevices = true;
      };
    };

    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
    };
  };
}
