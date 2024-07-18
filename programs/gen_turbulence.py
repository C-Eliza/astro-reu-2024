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
    not correlated (see https://doi.org/10.1093/mnras/stad2195), assuming
    relationships between the two as described by https://doi.org/10.1093/mnras/stad1631,
    and assuming Kolmogorov turbulence (power spectrum slope = 11/3 for density and 5/3 for
    velocity).

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

    # fractional density (density / mean_density) dispersion
    frac_n_turb = mach_number * driving_parameter

    # generate cubes
    frac_density = make_3dfield(
        imsize, powerlaw=11.0 / 3, amp=frac_n_turb, randomseed=seed
    )
    density = mean_density * (1.0 + frac_density)
    density[density < 0.0] = 0.0

    velocity = make_3dfield(
        imsize, powerlaw=5.0 / 3.0, amp=v_turb.to("km/s").value, randomseed=seed + 1
    )

    return density, velocity * u.km / u.s

def save_turbulence(density, velocity):
    hdu = fits.PrimaryHDU(velocity.to("km/s").value.T)
    hdu.writeto("debug/bigvelocity.fits", overwrite=True)
    hdu = fits.PrimaryHDU(density.to("cm**-3").value.T)
    hdu.writeto("debug/bigdensity.fits", overwrite=True)

if __name__ == "__main__":
    density, velocity = gen_turbulence(
            imsize=500,
            )
    save_turbulence(density,velocity)
