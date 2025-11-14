{
  description = "DoodleParty - Real-time collaborative drawing platform with AI-powered content moderation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Python environment with ML dependencies
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          tensorflow
          keras
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
          pytest
          pytest-cov
          black
          flake8
          mypy
          sphinx
          sphinx-rtd-theme
          rich
          # For training and optimization
          # tensorflow-model-optimization not in nixpkgs, install via pip if needed
        ]);

        # Node.js dependencies
        nodeDependencies = pkgs.mkYarnModules {
          pname = "doodleparty-node-modules";
          version = "1.0.0";
          packageJSON = ./package.json;
          yarnLock = ./yarn.lock;
        };

        # Build the frontend
        frontend = pkgs.buildNpmPackage {
          pname = "doodleparty-frontend";
          version = "1.0.0";

          src = ./.;

          npmDepsHash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="; # Update after first build

          buildPhase = ''
            npm run build
          '';

          installPhase = ''
            mkdir -p $out
            cp -r dist/* $out/
          '';
        };

      in
      {
        packages = {
          default = self.packages.${system}.doodleparty;

          doodleparty = pkgs.stdenv.mkDerivation {
            pname = "doodleparty";
            version = "1.0.0";

            src = ./.;

            buildInputs = [
              pythonEnv
              pkgs.nodejs_22
            ];

            installPhase = ''
              mkdir -p $out/bin
              mkdir -p $out/share/doodleparty

              # Copy Python backend
              cp -r src_py $out/share/doodleparty/
              cp requirements.txt $out/share/doodleparty/

              # Copy frontend build
              cp -r dist $out/share/doodleparty/public

              # Copy data and models
              cp -r data $out/share/doodleparty/
              cp -r models $out/share/doodleparty/

              # Create startup script
              cat > $out/bin/doodleparty <<EOF
              #!${pkgs.bash}/bin/bash
              cd $out/share/doodleparty
              export PYTHONPATH=$out/share/doodleparty:\$PYTHONPATH
              exec ${pythonEnv}/bin/python -m flask run --host=0.0.0.0 --port=\''${PORT:-5000}
              EOF
              chmod +x $out/bin/doodleparty
            '';
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pythonEnv
            nodejs_22
            yarn
            nodePackages.typescript
            nodePackages.typescript-language-server
            nodePackages.vscode-langservers-extracted
            python3Packages.python-lsp-server
            
            # Development tools
            git
            pre-commit
            
            # TensorFlow Lite runtime (for RPi4)
            # Note: May need custom build for ARM
          ];

          shellHook = ''
            echo "🎨 DoodleParty Development Environment"
            echo "======================================"
            echo "Python: $(python --version)"
            echo "Node.js: $(node --version)"
            echo "npm: $(npm --version)"
            echo ""
            echo "Available commands:"
            echo "  npm run dev       - Start Vite dev server"
            echo "  npm run build     - Build production frontend"
            echo "  npm run lint      - Run ESLint"
            echo "  npm run test      - Run tests"
            echo "  python -m pytest  - Run Python tests"
            echo "  scripts/training/train.py --help - Train model"
            echo "  scripts/data_processing/download_quickdraw_npy.py --help - Download data"
            echo ""
            
            # Set up Python path
            export PYTHONPATH="${./}:$PYTHONPATH"
            
            # Install npm dependencies if needed
            if [ ! -d "node_modules" ]; then
              echo "Installing npm dependencies..."
              npm install
            fi
          '';
        };

        # NixOS module
        nixosModules.default = import ./module.nix;
      }
    ) // {
      # NixOS module available at flake level
      nixosModules.doodleparty = import ./module.nix;
    };
}
