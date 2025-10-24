let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.numpy
      python-pkgs.astropy
      python-pkgs.scipy
      python-pkgs.tqdm
      python-pkgs.numba
      python-pkgs.pip
      python-pkgs.matplotlib
      python-pkgs.wheel
    ]))
  ];

  shellHook = ''
    python -m venv .venv
    source .venv/bin/activate
    pip install turbustat bettermoments
  '';
}
