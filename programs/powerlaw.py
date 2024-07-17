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
    if hdu.data.ndim == 4:
        data = np.sum(hdu.data[0],axis=0)
    elif hdu.data.ndim == 3:
        data = np.sum(hdu.data,axis=0)
    else:
        data = hdu.data
    pspec = PowerSpectrum(np.nan_to_num(data),header=hdu.header)
    pspec.run(xunit=u.arcsec**-1,verbose=True,low_cut=1/(15*u.arcmin))
    first = pspec.slope
    second = pspec.slope2D
    return(first, second)

"""
powerlaw1 = get_power_law("fits/region1rrl_200.0_M1.fits")
powerlaw2 = get_power_law("fits/region2rrl_200.0_M1.fits")

print(powerlaw1)
print(powerlaw2)
"""
