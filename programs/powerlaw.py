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
        data = np.sum(hdu.data[0][:,0:50,0:50],axis=0)
    elif hdu.data.ndim == 3:
        data = np.sum(hdu.data,axis=0)
    else:
        data = hdu.data
    pspec = PowerSpectrum(data,header=hdu.header)
    pspec.run(xunit=u.pix**-1)
    return pspec

r1 = get_power_law("fits/region1rrl_200.0_M1.fits")
r2 = get_power_law("fits/region2rrl_200.0_M1.fits")
print(round(r1.slope,2),round(r2.slope,2))
#print(get_power_law("~/Downloads/g320.channel.uvtaper.16stack.image.imsmooth.30arcsec.pbcor.line.fits"))

#velp = get_power_law("debug/bigvelocity.fits")
#densp = get_power_law("debug/bigdensity.fits")

#print("Velocity 1D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
#print("Velocity 2D:",round(velp.slope,2),"+-",round(velp.slope_err,2))
#print("Note: density powerlaw generated after subtracting mean from field")
#print("Density 1D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
#print("Density 2D:",round(densp.slope,2),"+-",round(densp.slope_err,2))
