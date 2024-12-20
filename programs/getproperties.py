"""
getproperties.py
Records properties of HII regions from RRL images and their moment maps    
Eliza Canales
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import bettermoments as bm
import argparse
import astropy.io.fits as fits
from fit_gradient import fit 

def main(filename):

    #Opening files
    print("Opening files")
    hdu = fits.open(filename)[0]
    hpbw = hdu.header['BMAJ'] * 3600
    data3d, velax = bm.load_cube(filename)
    moment0 = fits.open(filename.replace('.fits','_M0.fits'))[0].data
    e_moment0 = fits.open(filename.replace('.fits','_dM0.fits'))[0].data
    moment1 = fits.open(filename.replace('.fits','_M1.fits'))[0].data
    e_moment1 = fits.open(filename.replace('.fits','_dM1.fits'))[0].data
    moment2 = fits.open(filename.replace('.fits','_M2.fits'))[0].data
    e_moment2 = fits.open(filename.replace('.fits','_dM2.fits'))[0].data

    #Getting stastics
    print("Stddev")
    moment1std = np.nanstd(moment1)
    moment2std = np.nanstd(moment2)

    #Measuring "smoothness" of plane fit (higher = less smooth)
    print("residual")
    moment0residual = fit(moment0,e_moment0,data3d,velax,hdu.header)
    moment1residual = fit(moment1,e_moment1,data3d,velax,hdu.header)
    moment2residual = fit(moment2,e_moment2,data3d,velax,hdu.header)

    #Power spectrum
    print("pspec")
    pspec0 = PowerSpectrum(moment0, header=hdu.header)
    pspec1 = PowerSpectrum(moment1, header=hdu.header)
    pspec2 = PowerSpectrum(moment2, header=hdu.header)
    pspec0.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec1.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec2.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))

    print(filename,'stdM1',moment1std,'stdM2',moment2std,
          'resM0',moment0residual,'resM1',moment1residual,'resM2',moment2residual,
          'powerspec1DM0', pspec0.slope, 'e', pspec0.slope_err,
          'powerspec2DM0', pspec0.slope2D, 'e', pspec0.slope2D_err,
          'powerspec1DM1', pspec1.slope, 'e', pspec1.slope_err,
          'powerspec2DM1', pspec1.slope2D, 'e', pspec1.slope2D_err,
          'powerspec1DM2', pspec2.slope, 'e', pspec2.slope_err,
          'powerspec2DM2', pspec2.slope2D, 'e', pspec2.slope2D_err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates properties of HII images and moment maps",
        prog="getproperties.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        type=str,
        default="region1",
        help="Filename",
    )
    args = parser.parse_args()
    main(**vars(args))
