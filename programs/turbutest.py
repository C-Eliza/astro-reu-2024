"""
Desperate attempt to use fourier space to maintain power law properties when non-cube is used for turbulence.

Eliza Canales and Trey Wenger

Much code stolen from turbustat
"""
import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from HIIsim import generate_bool_sphere
from turbustat.simulator import make_3dfield
from turbustat.statistics import PowerSpectrum
from scipy.signal import convolve

imsize = 20
testfield1 = make_3dfield(imsize)
hdu = fits.PrimaryHDU(testfield1)
hdu.writeto('debug/controlfield.fits',overwrite=True)
testfield2 = make_3dfield(imsize, return_fft=True)
bool_sphere = 1.0 * generate_bool_sphere(radius=1.0*u.pc,
                                   impix=imsize,
                                   imsize=5*u.pc,
                                   lospix=imsize)
"""
fft_sphere = np.fft.fftshift(bool_sphere)
print(fft_sphere.shape)
sphered_field = np.fft.irfftn(fft_sphere * testfield2,(imsize,imsize,imsize))
"""
sphered_field = convolve(bool_sphere,testfield1)

hdu = fits.PrimaryHDU(sphered_field)
hdu.header["CDELT1"] = (1*u.arcsec/u.deg).value
hdu.header["CDELT2"] = (1*u.arcsec/u.deg).value 
hdu.header["CDELT3"] = (1*u.arcsec/u.deg).value
hdu.writeto('debug/spheredfield.fits',overwrite=True)

pspec1=PowerSpectrum(np.sum(testfield1,axis=2),header=hdu.header)
pspec1.run(xunit=u.pix**-1)
pspec2=PowerSpectrum(np.sum(sphered_field,axis=2),header=hdu.header)
pspec2.run(xunit=u.pix**-1)

print(pspec1.slope, pspec2.slope)
