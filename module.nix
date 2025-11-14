{ config, lib, pkgs, ... }:

with lib;

let
  cfg = config.services.doodleparty;
  
  pythonEnv = pkgs.python3.withPackages (ps: with ps; [
    tensorflow
    numpy
    opencv4
    scikit-learn
    pillow
    flask
    flask-cors
    gunicorn
    pandas
    matplotlib
    scipy
    python-dotenv
    pyyaml
    tqdm
  ]);

in {
  options.services.doodleparty = {
    enable = mkEnableOption "DoodleParty collaborative drawing platform";

    package = mkOption {
      type = types.package;
      default = pkgs.callPackage ./. { };
      defaultText = literalExpression "pkgs.doodleparty";
      description = "The DoodleParty package to use";
    };

    user = mkOption {
      type = types.str;
      default = "doodleparty";
      description = "User account under which DoodleParty runs";
    };

    group = mkOption {
      type = types.str;
      default = "doodleparty";
      description = "Group under which DoodleParty runs";
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Host address to bind to";
    };

    port = mkOption {
      type = types.port;
      default = 5000;
      description = "Port to listen on";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/doodleparty";
      description = "Directory for DoodleParty data and state";
    };

    modelsDir = mkOption {
      type = types.path;
      default = "/var/lib/doodleparty/models";
      description = "Directory for ML models";
    };

    logLevel = mkOption {
      type = types.enum [ "DEBUG" "INFO" "WARNING" "ERROR" "CRITICAL" ];
      default = "INFO";
      description = "Logging level for the application";
    };

    workers = mkOption {
      type = types.int;
      default = 4;
      description = "Number of Gunicorn worker processes";
    };

    maxConcurrentUsers = mkOption {
      type = types.int;
      default = 100;
      description = "Maximum number of concurrent users";
    };

    enableAI = mkOption {
      type = types.bool;
      default = true;
      description = "Enable AI-powered content moderation";
    };

    enableLLM = mkOption {
      type = types.bool;
      default = false;
      description = "Enable LLM integration for prompts and narration";
    };

    llmApiKey = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "API key for LLM service (DigitalOcean AI)";
    };

    llmApiKeyFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "File containing the API key for LLM service";
    };

    moderationThreshold = mkOption {
      type = types.float;
      default = 0.85;
      description = "Threshold for content moderation (0.0 - 1.0)";
    };

    enableMetrics = mkOption {
      type = types.bool;
      default = true;
      description = "Enable Prometheus metrics endpoint";
    };

    extraConfig = mkOption {
      type = types.attrs;
      default = {};
      description = "Extra configuration options as attribute set";
      example = literalExpression ''
        {
          CANVAS_WIDTH = 1920;
          CANVAS_HEIGHT = 1080;
          SAVE_INTERVAL = 60;
        }
      '';
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open the firewall for the configured port";
    };
  };

  config = mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.dataDir;
      createHome = true;
      description = "DoodleParty service user";
    };

    users.groups.${cfg.group} = {};

    systemd.services.doodleparty = {
      description = "DoodleParty - Real-time collaborative drawing platform";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        PYTHONPATH = "${cfg.dataDir}/src-py";
        FLASK_APP = "core.app";
        HOST = cfg.host;
        PORT = toString cfg.port;
        DATA_DIR = cfg.dataDir;
        MODELS_DIR = cfg.modelsDir;
        LOG_LEVEL = cfg.logLevel;
        MAX_CONCURRENT_USERS = toString cfg.maxConcurrentUsers;
        ENABLE_AI = if cfg.enableAI then "true" else "false";
        ENABLE_LLM = if cfg.enableLLM then "true" else "false";
        MODERATION_THRESHOLD = toString cfg.moderationThreshold;
        ENABLE_METRICS = if cfg.enableMetrics then "true" else "false";
      } // (mapAttrs (name: value: toString value) cfg.extraConfig);

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        
        # Load LLM API key from file if specified
        EnvironmentFile = mkIf (cfg.llmApiKeyFile != null) cfg.llmApiKeyFile;
        
        # Security hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [ "AF_UNIX" "AF_INET" "AF_INET6" ];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        LockPersonality = true;
        SystemCallFilter = [ "@system-service" "~@privileged" ];
        
        # Resource limits
        LimitNOFILE = 65536;
        
        # Restart policy
        Restart = "on-failure";
        RestartSec = "10s";
        
        # Startup command using Gunicorn
        ExecStart = ''
          ${pythonEnv}/bin/gunicorn \
            --bind ${cfg.host}:${toString cfg.port} \
            --workers ${toString cfg.workers} \
            --worker-class gevent \
            --timeout 120 \
            --access-logfile - \
            --error-logfile - \
            core.app:app
        '';
      };

      preStart = ''
        # Ensure directories exist
        mkdir -p ${cfg.dataDir}/data
        mkdir -p ${cfg.modelsDir}
        mkdir -p ${cfg.dataDir}/logs
        
        # Copy application files if not present
        if [ ! -d "${cfg.dataDir}/src-py" ]; then
          cp -r ${cfg.package}/share/doodleparty/src-py ${cfg.dataDir}/
        fi
        
        # Copy models if available
        if [ -d "${cfg.package}/share/doodleparty/models" ] && [ ! -f "${cfg.modelsDir}/model.tflite" ]; then
          cp -r ${cfg.package}/share/doodleparty/models/* ${cfg.modelsDir}/ || true
        fi
        
        # Set permissions
        chown -R ${cfg.user}:${cfg.group} ${cfg.dataDir}
        chmod 750 ${cfg.dataDir}
      '';
    };

    # Nginx reverse proxy configuration (optional)
    services.nginx = mkIf (config.services.nginx.enable) {
      virtualHosts."doodleparty" = mkIf cfg.enable {
        locations."/" = {
          proxyPass = "http://${cfg.host}:${toString cfg.port}";
          proxyWebsockets = true;
          extraConfig = ''
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            
            # Timeouts for long-lived connections
            proxy_connect_timeout 7d;
            proxy_send_timeout 7d;
            proxy_read_timeout 7d;
          '';
        };
      };
    };

    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
    };
  };

  meta = {
    maintainers = with maintainers; [ ];
    doc = ./README.md;
  };
}
