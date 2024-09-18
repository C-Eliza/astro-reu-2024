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
    hdu.writeto("debug/bigvelocity.fits", overwrite=True)
    hdu = fits.PrimaryHDU(density.to("cm**-3").value.T)
    hdu.writeto("debug/bigdensity.fits", overwrite=True)

def turbstitch(
    imsize,
    mach_number=1.0,
    mean_density=200.0 / u.cm**3,
    driving_parameter=0.75,
    seed=1234,
    num=5,
):
    """
    Generates a turbulence prism by generating several data cubes and putting them back to back.

    Inputs:
        Same as gen_turbulence
        num -- How many times deeper the final prism is compared to the pixel length

    Outputs:
        density -- Density prism
        velocity -- Velocity prism
    """
    density, velocity = gen_turbulence(
        imsize,
        mach_number=mach_number,
        mean_density=mean_density,
        driving_parameter=driving_parameter,
        seed=seed,
        )
    for i in range(num-1):
        tempdensity, tempvelocity = gen_turbulence(
            imsize,
            mach_number=mach_number,
            mean_density=mean_density,
            driving_parameter=driving_parameter,
            seed=seed+i+1,
            )
        density = np.append(density, tempdensity, axis=2)
        velocity = np.append(velocity, tempvelocity, axis=2)

    return(density,velocity)

def gen_low_resolution(
        imsize=50,
        mean_density=1000*u.cm**-3,
        seed=100,
        mach_number=5,
        driving_parameter=0.75,
        num=5,
        ):
    """
    Generates a turbulence prism by generating a large data cube with sides the length of the depth.
    Reduces the resolution of each 2d layer to shape into a prism.

    Inputs:
        Same as gen_turbulence
        num -- How many times deeper the final prism is compared to the pixel length

    Outputs:
        lowdensity -- Density prism
        lowvelocity -- Velocity prism
    """
    density, velocity = gen_turbulence(
        imsize*num,
        mach_number=mach_number,
        mean_density=mean_density,
        driving_parameter=driving_parameter,
        seed=seed,
        )
    # Initializing low resolution versions of density and velocity cubes
    lowdensity = np.zeros((imsize,imsize,num*imsize))*u.cm**-3
    lowvelocity = np.zeros((imsize,imsize,num*imsize))*u.km/u.s
    """
    # Mean case
    for xpixel in range(imsize):
        for ypixel in range(imsize):
            lowdensity[xpixel,ypixel] = np.mean(density[num*xpixel:num*(xpixel+1),num*ypixel:num*(ypixel+1)])
            lowvelocity[xpixel,ypixel] = np.mean(velocity[num*xpixel:num*(xpixel+1),num*ypixel:num*(ypixel+1)])
    """
    # Mean squared, rooted case
    for xpixel in range(imsize):
        for ypixel in range(imsize):
            lowdensity[xpixel,ypixel] = np.sqrt(np.mean(density[num*xpixel:num*(xpixel+1),num*ypixel:num*(ypixel+1)]**2))
            lowvelocity[xpixel,ypixel] = np.sqrt(np.mean(velocity[num*xpixel:num*(xpixel+1),num*ypixel:num*(ypixel+1)]**2))
    
    return lowdensity, lowvelocity

if __name__ == "__main__":
    density, velocity = gen_low_resolution(
            imsize=50,
            mean_density=1000*u.cm**-3,
            seed=100,
            mach_number=5,
            driving_parameter=0.75,
            num=1,
            )
    save_turbulence(density,velocity)
