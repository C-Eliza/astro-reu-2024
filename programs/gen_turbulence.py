"""
gen_turbulence.py
Generate tubulent density and velocity fields.
Trey Wenger & Eliza Canales - July 2024
"""

import numpy as np
import astropy.units as u
from astropy.io import fits

from turbustat.simulator import make_3dfield


def gen_turbulence(
    imsize,
    mach_number=1.0,
    mean_density=200.0 / u.cm**3,
    driving_parameter=0.75,
    seed=1234,
):
    """
    Generate a turbulent electron density and radial velocity field, assuming the two are
    not correlated (see https://doi.org/10.1093/mnras/stad2195), assuming the velocity
    is described by https://doi.org/10.1093/mnras/stad1631, the density 
    https://arxiv.org/pdf/1206.4524 and assuming Kolmogorov turbulence (power spectrum 
    slope = 11/3 for density and 5/3 for velocity).

    Inputs:
        imsize :: integer
            Simulated cube size
        mach_number :: scalar
            Mach number. <1 = subsonic, >1 = supersonic.
        mean_density :: scalar (with units)
            Mean electron density
        driving_parameter :: scalar
            Driving parameter. =0.33 for solenoidal driving and =1.0 for compressive driving
        seed :: scaslar
            Random seed

    Returns:
        density :: 3-D array of scalars (with units)
            Density field
        velocity :: 3-D array of scalars (with units)
            Radial velocity field
    """
    # ionized gas sound speed (from https://doi.org/10.1093/mnras/stad2195)
    c_s = 11.0 * u.km / u.s

    # 1D turbulent velocity dispersion
    v_turb = mach_number * c_s / np.sqrt(3.0)

    # logarithmic density (density / mean_density) dispersion
    log_n_turb = np.sqrt(np.log(1 + driving_parameter**2 * mach_number**2))

    # generate cubes
    log_density_frac = make_3dfield(
        imsize, powerlaw=11.0 / 3, amp=log_n_turb, randomseed=seed
    )
    density = np.exp(np.log(mean_density / u.cm**-3) + log_density_frac) * u.cm**-3

    velocity = make_3dfield(
        imsize, powerlaw=5.0 / 3.0, amp=v_turb.to("km/s").value, randomseed=seed + 1
    )

    return density, velocity * u.km / u.s

def save_turbulence(density, velocity):
    """
    Writes to a file for debugging
    """
    hdu = fits.PrimaryHDU(velocity.to("km/s").value.T)
    hdu.writeto("debug/velocity.fits", overwrite=True)
    hdu = fits.PrimaryHDU(density.to("cm**-3").value.T)
    hdu.writeto("debug/density.fits", overwrite=True)
