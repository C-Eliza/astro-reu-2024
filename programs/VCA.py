"""
VCA.py
Calculate the VCA for a given radio image.
Eliza Canales and Trey Wenger - Summer 2024
"""

import numpy as np
from turbustat.statistics import PowerSpectrum, VCA, VCS, SCF
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import argparse

def get_VCA(filename):
    """
    Takes a fits file and returns its VCA

    Inputs:
        filename -- Name of the fits file

    Returns:
        vspec -- Object that describes the VCA
    """
    hdu = fits.open(filename)[0]
    vspec = VCA(hdu.data,header=hdu.header)
    vspec.run(xunit=u.arcsec**-1,high_cut=1/(hdu.header['BMAJ']*u.deg))
    return vspec

def main(fnames):
    impix = [int(fname.split("_")[1].replace("rrl", "")) for fname in fnames]
    VCA_slope = []
    VCA_slopeu = []

    for fname, pix in zip(fnames, impix):
        VCAres = get_VCA(fname)
        VCA_slope.append(VCAres.slope)
        VCA_slopeu.append(VCAres.slope_err)

    for i in range(len(impix)):
        print("Simulated 1D w/",impix[i],"pixels:",VCA_slope[i],"+-",VCA_slopeu[i])

#print("Real 1D:",round(real.slope,2),"+-",round(real.slope_err,2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate velocity statistics",
        prog="VLSR.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fnames",
        type=str,
        nargs="+",
        help="RRL data cubes",
    )
    args = parser.parse_args()
    main(**vars(args))
"""
velp = get_power_law("debug/bigvelocity.fits")
densp = get_power_law("debug/bigdensity.fits")

print("Velocity 1D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
print("Velocity 2D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
print("Density 1D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
print("Density 2D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
"""
