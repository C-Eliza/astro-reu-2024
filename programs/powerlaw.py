"""
powerlaw.py
Calculate the powerlaw for a given radio image.
Eliza Canales and Trey Wenger - Summer 2024
"""

import numpy as np
from turbustat.statistics import PowerSpectrum
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt

def get_power_law(filename):
    """
    Takes a fits file and returns its power spectrum object

    Inputs:
        filename -- Name of the fits file

    Returns:
        pspec -- Object that describes the power spectrum
    """
    hdu = fits.open(filename)[0]
    pspec = PowerSpectrum(hdu.data,header=hdu.header)
    pspec.run(xunit=u.pix**-1,low_cut=1/(50*u.pix)) #Arbitary low cut
    return pspec

#r1 = get_power_law("fits/region1rrl_200.0_M1.fits")
r2 = get_power_law("sim/region1rrlsim_M1.fits")
#real = get_power_law("~/Downloads/hii_data/g320.channel.uvtaper.16stack.image.imsmooth.30arcsec.pbcor.vlsr.fits")
print(round(r2.slope,3))
"""
velp = get_power_law("debug/bigvelocity.fits")
densp = get_power_law("debug/bigdensity.fits")

print("Velocity 1D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
print("Velocity 2D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
print("Density 1D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
print("Density 2D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
"""
