# Shell.nix - for non-flake development environments
# Usage: nix-shell

{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  }
}:

let
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
  ]);

in pkgs.mkShell {
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
    
    # Optional: TensorFlow Lite runtime
    # Note: May need custom build for ARM/RPi4
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
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Install npm dependencies if needed
    if [ ! -d "node_modules" ]; then
      echo "Installing npm dependencies..."
      npm install
    fi
  '';
}
