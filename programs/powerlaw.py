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
    hdu = fits.open(filename)[0]
    print(hdu.shape)
    if hdu.data.ndim == 4:
        moment0 = np.sum(hdu.data[0],axis=0)
    else:
        moment0 = np.sum(hdu.data,axis=0)
    pspec = PowerSpectrum(np.nan_to_num(moment0),header=hdu.header)
    pspec.run(xunit=u.pix**-1)
    first = pspec.slope
    second = pspec.slope2D
    return(first, second)

print(get_power_law("fits/region1rrl_200.0.fits"))
print(get_power_law("fits/region2rrl_200.0.fits"))
print(get_power_law("~/Downloads/g320.channel.uvtaper.16stack.image.imsmooth.30arcsec.pbcor.line.fits"))
