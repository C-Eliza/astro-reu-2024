{ pkgs ? import <nixpkgs> {} }:
let
    turbustat = 
      pkgs.python3Packages.buildPythonPackage rec {
        format = "pyproject";
        build-system = with pkgs.python3Packages; [ setuptools ];
        pname = "turbustat";
        version = "v1.4.dev5+gc837780";
        src = pkgs.fetchFromGitHub {
          owner = "Astroua";
          rev = "adac77bd17ae5b6c74705a997fa8e7d5b26d5b9e";
          repo = "TurbuStat";
          hash = "sha256-OoVFkoRkDGE2R9OIc28xNvT9sCe3MtTvqZCObAC6nOk=";
        };
        nativeBuildInputs = with pkgs.python3Packages; [
          oldest-supported-numpy
          extension-helpers
          cython
          setuptools-scm
        ];
        propagatedBuildInputs = with pkgs.python3Packages; [
          astropy
          matplotlib
          scipy
          scikit-learn
          statsmodels
          scikit-image
          spectral-cube
        ];
      };

    zeus-mcmc = pkgs.python3Packages.buildPythonPackage rec {
        format = "pyproject";
        build-system = with pkgs.python3Packages; [ setuptools ];
        pname = "zeus-mcmc";
        version = "2.5.4";
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-WUuqkN5K1EiMTbXtagRG9xA7xLPeeH9NfSPJHJqoh2k=";
        };
        pythonRemoveDeps = [
          "argparse"
        ];
        propagatedBuildInputs = with pkgs.python3Packages; [
          numpy
          scipy
          tqdm
          pytest
          matplotlib
          seaborn
          scikit-learn
        ];
    };

    bettermoments = pkgs.python3Packages.buildPythonPackage rec {
        format = "pyproject";
        build-system = with pkgs.python3Packages; [ setuptools ];
        pname = "bettermoments";
        version = "1.10.0";
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-q7dJpOa7mrsekSFm05jgkCfEovy9JpI0NxzELr2AuTE=";
        };
        pythonRemoveDeps = [
          "argparse"
        ];
        propagatedBuildInputs = with pkgs.python3Packages; [
          numpy
          astropy
          tqdm
          emcee
          zeus-mcmc
        ];
    };
in
pkgs.mkShell{
  nativeBuildInputs = [
    (
      let 
      my-python-packages = p: with p; [
        setuptools
        numpy
        astropy
        scipy
        tqdm
        numba
        pip
        matplotlib
        turbustat
        bettermoments
      ]; in pkgs.python3.withPackages my-python-packages
    )      
  ];
}
