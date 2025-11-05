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
      numpy
      pandas
      matplotlib
      scikit-learn
      pillow
      flask
      flask-cors
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
      default = "127.0.0.1";
      description = "Host address to bind to";
    };

    modelPath = mkOption {
      type = types.path;
      description = "Path to the trained model file";
      example = "/var/lib/doodlehunter/models/quickdraw_classifier.h5";
    };

    dataDir = mkOption {
      type = types.path;
      default = "/var/lib/doodlehunter";
      description = "Directory for DoodleHunter data";
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

    systemd.services.doodlehunter = {
      description = "DoodleHunter Web Interface";
      after = [ "network.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        MODEL_PATH = cfg.modelPath;
        FLASK_PORT = toString cfg.port;
        FLASK_HOST = cfg.host;
        PYTHONPATH = "${doodlehunter}/${python.sitePackages}";
      };

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        WorkingDirectory = cfg.dataDir;
        ExecStart = "${python}/bin/python ${doodlehunter}/${python.sitePackages}/src/web/app.py";
        Restart = "on-failure";
        RestartSec = "5s";

        # Security hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.dataDir ];
      };
    };

    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [ cfg.port ];
    };
  };
}
