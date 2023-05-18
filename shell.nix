/*
  A nix-shell configuration for running this project on Macs with Apple CPUS.

  To launch a shell in this environment:
    nix-shell
*/

{ pkgs ? (import <nixpkgs> {})}: 

let 
  /* cannot detect if using apple vs intel hardware directly,
     but installation is the same for intel CPU and apple Metal GPU versions
     anyways, so doesnt matter */
  isMac = (pkgs.system == "aarch64-darwin");

in
  assert pkgs.lib.asserts.assertMsg (isMac) ''
    This shell.nix file currently supports macs only.
    Consider using the dockerfile for NVIDIA systems.
    ''; 

    pkgs.mkShell {
      packages = [pkgs.python39 pkgs.neovim];
      shellHook = ''
        export VIRTUAL_ENV_DISABLE_PROMPT=1
        rm -rf .venv
        python3 -m venv .venv
        . .venv/bin/activate
        pip install -r requirements.lock.txt
        pip install -e .
      '';
    }
