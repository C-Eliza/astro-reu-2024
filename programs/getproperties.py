"""
getproperties.py
Records properties of HII regions from RRL images and their moment maps    
Eliza Canales
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import argparse
import astropy.io.fits as fits
from turbustat.statistics import PowerSpectrum

def main(filename):

    #Opening files
    print("Opening files")
    hdu = fits.open(filename)[0]
    hpbw = hdu.header['BMAJ'] * 3600
    moment0 = fits.open(filename.replace('.fits','_M0.fits'))[0].data
    moment1 = fits.open(filename.replace('.fits','_M1.fits'))[0].data
    moment2 = fits.open(filename.replace('.fits','_M2.fits'))[0].data

    #Power spectrum
    print("pspec")
    pspec0 = PowerSpectrum(moment0, header=hdu.header)
    pspec1 = PowerSpectrum(moment1, header=hdu.header)
    pspec2 = PowerSpectrum(moment2, header=hdu.header)
    pspec0.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec1.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec2.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec0.save_results(filename.replace('.fits','_M0'))
    pspec1.save_results(filename.replace('.fits','_M1'))
    pspec2.save_results(filename.replace('.fits','_M2'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates the powerlaw spectrum and saves it",
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
