# Default.nix - for non-flake users
# This provides a simple way to build DoodleParty without flakes

{ pkgs ? import <nixpkgs> {
    config.allowUnfree = true;
  }
}:

let
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
    pytest
    pytest-cov
    black
    flake8
    mypy
    sphinx
    sphinx-rtd-theme
    rich
  ]);

in pkgs.stdenv.mkDerivation {
  pname = "doodleparty";
  version = "1.0.0";

  src = ./.;

  buildInputs = [
    pythonEnv
    pkgs.nodejs_22
    pkgs.yarn
  ];

  buildPhase = ''
    # Install npm dependencies
    export HOME=$TMPDIR
    npm install
    
    # Build frontend
    npm run build
  '';

  installPhase = ''
    mkdir -p $out/bin
    mkdir -p $out/share/doodleparty

    # Copy Python backend
    cp -r src-py $out/share/doodleparty/
    cp requirements.txt $out/share/doodleparty/

    # Copy frontend build
    cp -r dist $out/share/doodleparty/public

    # Copy data and models
    cp -r data $out/share/doodleparty/ || true
    cp -r models $out/share/doodleparty/ || true

    # Create startup script
    cat > $out/bin/doodleparty <<EOF
    #!${pkgs.bash}/bin/bash
    cd $out/share/doodleparty
    export PYTHONPATH=$out/share/doodleparty:\$PYTHONPATH
    exec ${pythonEnv}/bin/python -m flask run --host=0.0.0.0 --port=\''${PORT:-5000}
    EOF
    chmod +x $out/bin/doodleparty
  '';

  meta = with pkgs.lib; {
    description = "Real-time collaborative drawing platform with AI-powered content moderation";
    homepage = "https://github.com/yourusername/doodleparty";
    license = licenses.gpl3;
    platforms = platforms.unix;
    maintainers = [];
  };
}
