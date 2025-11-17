{ pkgs ? import <nixpkgs> {} }:
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

        (
          buildPythonPackage rec {
            format = "pyproject";
            build-system = [ setuptools ];
            pname = "turbustat";
            version = "v1.4.dev5+gc837780";
            src = pkgs.fetchFromGitHub {
              owner = "Astroua";
              rev = "adac77bd17ae5b6c74705a997fa8e7d5b26d5b9e";
              repo = "TurbuStat";
              hash = "sha256-OoVFkoRkDGE2R9OIc28xNvT9sCe3MtTvqZCObAC6nOk=";
            };
            nativeBuildInputs = [
              oldest-supported-numpy
              extension-helpers
              cython
              setuptools_scm
            ];
            propagatedBuildInputs = [
              astropy
              matplotlib
              scipy
              scikit-learn
              statsmodels
              scikit-image
              spectral-cube
            ];
          }
        )

        (
          buildPythonPackage rec {
            format = "pyproject";
            build-system = [ setuptools ];
            pname = "bettermoments";
            version = "1.9.2";
            src = fetchPypi {
              inherit pname version;
              sha256 = "sha256-Zng9AP5uKBM1mdzCLH9ZTB+6SkqY3VvQtRl5eM5g6mE=";
            };
            pythonRemoveDeps = [
              "argparse"
            ];
            propagatedBuildInputs = [
              numpy
              astropy
              tqdm
              emcee
              
              (
                buildPythonPackage rec {
                  format = "pyproject";
                  build-system = [ setuptools ];
                  pname = "zeus-mcmc";
                  version = "2.5.4";
                  src = fetchPypi {
                    inherit pname version;
                    sha256 = "sha256-WUuqkN5K1EiMTbXtagRG9xA7xLPeeH9NfSPJHJqoh2k=";
                  };
                  pythonRemoveDeps = [
                    "argparse"
                  ];
                  propagatedBuildInputs = [
                    numpy
                    scipy
                    tqdm
                    pytest
                    matplotlib
                    seaborn
                    scikit-learn
                  ];
                }
              )
            ];
          }
        )
      ]; in pkgs.python3.withPackages my-python-packages
    )      
  ];
}
