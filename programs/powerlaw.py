"""
powerlaw.py
Calculate the powerlaw for a given radio image.
Eliza Canales and Trey Wenger - Summer 2024
"""

import numpy as np
from turbustat.statistics import PowerSpectrum
import bettermoments as bm
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import argparse

def get_power_law(filename,moment,verbose):
    """
    Takes a fits file and returns its power spectrum object

    Inputs:
        filename -- Name of the fits file

    Returns:
        pspec -- Object that describes the power spectrum
    """
    hdu = fits.open(filename)[0]
    data, velax = bm.load_cube(filename)
    rms = bm.estimate_RMS(data, N=5)
    mask = bm.get_threshold_mask(data=data,clip=20.0)
    if(moment==0):
        momentmap = bm.collapse_zeroth(velax=velax, data=mask*data, rms=rms)
    elif(moment==1):
        momentmap = bm.collapse_first(velax=velax, data=mask*data, rms=rms)
    elif(moment==2):
        momentmap = bm.collapse_second(velax=velax, data=mask*data, rms=rms)

    pspec = PowerSpectrum(momentmap, header=hdu.header)
    pspec.run(xunit=u.arcsec**-1,high_cut=1/(hdu.header['BMAJ']*u.deg),verbose=verbose) #High cut based on beam

    return pspec

def main(fnames, moment, verbose):
    impix = [int(fname.split("_")[1].replace("rrl", "")) for fname in fnames]
    PS_slope = []
    PS_slopeu = []
    PS2_slope = []
    PS2_slopeu = []

    for fname, pix in zip(fnames, impix):
        PSres = get_power_law(fname,moment,verbose)
        PS_slope.append(PSres.slope)
        PS_slopeu.append(PSres.slope_err)
        PS2_slope.append(PSres.slope2D)
        PS2_slopeu.append(PSres.slope2D_err)

    for i in range(len(impix)):
        print("1D power law moment", moment,"w/",impix[i],"pixels:",PS_slope[i],"+-",PS_slopeu[i])
    print()
    for i in range(len(impix)):
        print("2D power law moment", moment,"w/",impix[i],"pixels:",PS2_slope[i],"+-",PS2_slopeu[i])
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate powerlaw statistics",
        prog="powerlaw.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fnames",
        type=str,
        nargs="+",
        help="RRL data cubes",
    )
    parser.add_argument(
        "-M",
        "--moment",
        type=int,
        default=0,
        help="Moment used in calculation"
    )
    parser.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Show slope and residuals chart"
    )
    args = parser.parse_args()
    main(**vars(args))
