{
  description = "DoodleHunter - Binary classification for drawing content moderation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python311;
        pythonPackages = python.pkgs;
      in
      {
        packages = {
          default = pythonPackages.buildPythonPackage {
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
              tqdm
              requests
              seaborn
            ];

            # Skip tests during build
            doCheck = false;

            meta = with pkgs.lib; {
              description = "Binary classification for drawing content moderation";
              homepage = "https://github.com/yourusername/doodlehunter";
              license = licenses.mit;
              maintainers = [ ];
            };
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python
            pythonPackages.pip
            pythonPackages.virtualenv
            pythonPackages.setuptools
            pythonPackages.wheel
          ] ++ (with pythonPackages; [
            tensorflow
            keras
            numpy
            pandas
            matplotlib
            scikit-learn
            pillow
            flask
            flask-cors
            tqdm
            requests
            seaborn
            pytest
            pytest-cov
            black
            flake8
            mypy
          ]);

          shellHook = ''
            echo "🎨 DoodleHunter Development Environment"
            echo ""
            echo "Available commands:"
            echo "  python scripts/train.py          - Train model"
            echo "  python scripts/evaluate.py       - Evaluate model"
            echo "  python src/web/app.py            - Start web interface"
            echo ""
            echo "Python version: $(python --version)"
            echo "TensorFlow available: $(python -c 'import tensorflow; print(tensorflow.__version__)' 2>/dev/null || echo 'Not installed')"
            echo ""
            
            # Set up Python path
            export PYTHONPATH="${self}/src:$PYTHONPATH"
          '';
        };

        apps = {
          default = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/doodlehunter";
          };

          train = {
            type = "app";
            program = "${pkgs.writeShellScript "train" ''
              ${python}/bin/python ${./scripts/train.py} "$@"
            ''}";
          };

          evaluate = {
            type = "app";
            program = "${pkgs.writeShellScript "evaluate" ''
              ${python}/bin/python ${./scripts/evaluate.py} "$@"
            ''}";
          };

          web = {
            type = "app";
            program = "${pkgs.writeShellScript "web" ''
              ${python}/bin/python ${./src/web/app.py} "$@"
            ''}";
          };
        };
      }
    );
}
