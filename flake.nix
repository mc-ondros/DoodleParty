# Root flake.nix - imports from infra/nix/
# This stub allows tools to find the flake at repo root while keeping
# Nix configuration organized under infra/nix/

{
  description = "DoodleParty - Collaborative drawing game with ML content moderation";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            nodejs
            python3
            python3Packages.pip
            git
          ];
        };
      }
    );
}
