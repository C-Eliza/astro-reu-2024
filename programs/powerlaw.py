"""
powerlaw.py
Calculate the powerlaw for a given radio image.
Eliza Canales and Trey Wenger - Summer 2024
"""

import numpy as np
from turbustat.statistics import PowerSpectrum
from astropy.io import fits
import astropy.units as u

fits.open("fits/testregion3d_200.fits")[0]
moment0 = np.sum(.data,axis=2)
pspec = PowerSpectrum(moment0,header=)
pspec.run(verbose=True,xunit=u.pix**-1)
