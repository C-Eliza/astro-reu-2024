"""
HIIsim.py
Simulate observations of radio continuum and radio recombination line emission
from Galactic HII regions.
Eliza Canales & Trey Wenger - Summer 2024
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from gen_turbulence import gen_turbulence
from gen_turbulence import turbstitch


def thermal_fwhm(temp, nu_0):
    """
    Calculate the FWHM line width due to thermal broadening.

    Inputs:
        temp -- Temperature of the hydrogen (with units)
        nu_0 -- Transition frequency (with units)

    Returns:
        fwhm -- Thermal FWHM line width (with units)
    """
    return np.sqrt(8 * np.log(2) * c.k_B / c.c**2) * np.sqrt(temp / c.m_p) * nu_0


def doppler(nu_0, nu):
    """
    Calculate the non-relativistic Doppler velocity.

    Inputs:
        nu_0 -- Rest frequency (with units)
        nu -- Observed frequency (with units)

    Returns:
        vel -- Doppler velocity (with units)
    """
    return c.c * (nu - nu_0) / nu_0


def doppler_freq(nu_0, vel):
    """
    Calculate the frequencies of non-relativistic Doppler shift.

    Inputs:
        nu_0 -- Rest frequency (with units)
        vel -- Doppler velocity (with units)

    Returns:
        nu -- Observed frequency (with units)
    """
    return nu_0 * np.sqrt((1 - vel / c.c) / (1 + vel / c.c))


def emission_measure(density, depth):
    """
    Calculate the emission measure, assuming a homogenous medium.
    Equation 4.57 from Condon & Ransom ERA textbook.

    TODO: generalize this for the non-homogenous case

    Inputs:
        density -- Electron number density (with units)
        depth -- Line-of-sight depth (with units)

    Alternatively:
        density -- 3d array of density (with units)
        depth -- Depth of each voxel (with units)

    Returns:
        em -- Emission measure 2d array (with units)
        or
        em -- Emission measure 3d array (with units)
    """
    return density**2 * depth


def ff_opacity(temp, nu, em):
    """
    Calculate the free-free opacity. Equation 4.60 from Condon & Ransom ERA textbook.

    Inputs:
        temp -- Electron temperature (with units)
        nu   -- 1-D array of frequencies at which to evaluate the opacity (with units)
        em   -- 2-D array of emission measures (with units)

    Returns:
        opacity -- 3-D array (shape em.shape + nu.shape) of free-free opacities (unitless)
    """
    return (
        (
            3.28e-7
            * (temp / (1e4 * u.K)) ** -1.35
            * (nu / u.GHz) ** -2.1
            * (em[..., None] / (u.pc * u.cm**-6))
        )
        .to("")
        .value
    )


def gaussian(x, amp, center, fwhm):
    """
    Evaluate a Gaussian.

    Inputs:
        x -- Position(s) at which to evaluate (with units)
        amp -- amplitude (with units)
        center -- centroid; same units as x (with units)
        fwhm -- FWHM width; same units as x (with units)

    Returns:
        y -- Gaussian evaluated at x
    """
    return amp * np.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)


def rrl_opacity(temp, em, nu, center_freq, fwhm_freq):
    """
    Calculate the radio recombination line opacity. Equation 7.96 from Condon & Ransom ERA textbook.

    Inputs:
        temp -- Electron temperature (with units)
        em   -- 3-D array of emission measures (with units)
        nu   -- Frequency to evalute at (with units)
        center_freq -- 3-D array of line center frequency (with units)
        fwhm_freq -- FWHM line width in frequency units (with units)

    Returns:
        opacity -- 3-D array (shape em.shape) of RRL opacities (unitless)
    """
    # Line center opacity (eq. 7.96)
    tau_center = (
        1.92e3
        * (temp / u.K) ** (-5.0 / 2.0)
        * (em / (u.pc * u.cm**-6))
        / (fwhm_freq / u.kHz)
    )
    return (
        gaussian(
            nu, tau_center, center_freq, fwhm_freq
        )
        .to("")
        .value
    )


def brightness_temp(optical_depth, temp):
    """
    Calculate the brightness temperature.

    Inputs:
        optical_depth -- Optical depth (unitless)
        temp -- Electron temperature (with units)

    Returns:
        TB -- brightness temperature (with units)
    """
    return temp * (1 - np.exp(-optical_depth))


def spatial_smooth(data_cube, radius):
    """
    Smooth a data cube along the spatial axes using a Gaussian filter.

    TODO: We typically think of "telescope beams" as the FWHM of the beam rather than
    the standard deviation. Too, we usually think of the beam in terms of an angular
    scale instead of a pixel scale. This function should probably take as input the FWHM
    of the smoothing kernel in angular units and then convert that to a standard deviation
    in pixel units to be supplied to gaussian_filter.

    Inputs:
        data_cube -- 3-D array of data to smooth. The first two axes are the spatial axes.
        radius    -- Standard deviation of the gaussian kernel (in pixels)

    Returns:
        smooth_data_cube -- 3-D array of spatially-smoothed data
    """
    # gaussian_filter does not support units
    unit = data_cube.unit
    return gaussian_filter(data_cube.value, [radius, radius, 0]) * unit

def generate_bool_sphere(radius,impix,imsize,lospix):
    """
    Creates a 3d array of Trues and Falses to represent whether a sphere exists
    in that part of a cube with the sphere's center about the middlemost
    part of the cube.

    Inputs:
        radius -- Radius of sphere to be generated (in units)
        impix -- Number of pixels in front face of cube
        imsize -- Physical size of image (in units)
        lospix -- Number of pixels in line of sight direction

    Returns:
        bool_sphere -- Matrix mapping where sphere exists
    """

    # Edges of grid cells
    grid_edges = np.linspace(-imsize/2, imsize/2, impix + 1, endpoint=True)
    los_edges = np.linspace(-imsize/2, imsize/2, lospix + 1, endpoint=True)

    # Centroids of grid cells
    grid_centers = grid_edges[:-1] + (grid_edges[1:] - grid_edges[:-1]) / 2.0
    los_centers = los_edges[:-1] + (los_edges[1:] - los_edges[:-1]) / 2.0

    # Cubic 3D grid of centroids
    gridX, gridY, gridZ = np.meshgrid(grid_centers, grid_centers, los_centers, indexing="ij")

    # The boolean sphere
    bool_sphere = (radius**2 > gridX**2 + gridY**2 + gridZ**2)
    return bool_sphere


def observe(physfile,filename,beam_fwhm,noise):
    """
    Takes in a simulated region, then simulates an observation of it. Saves to a
    FITS file.

    Inputs:
        physfile -- Filename of the simulated region
        filename -- Filename to have observation saved to
        beam_fwhm -- FWHM beam size (with units)
        noise -- Gaussian noise added to the synthetic data cube (brightness temperature; with units)

    Returns:
        Nothing
    """
    hdul = fits.open(f"sim/{physfile}")
    hdr = hdul[0].header
    TB = hdul[0].data.T * u.K
    pixel_size = hdr["CDELT1"] * u.deg
    hdul.close()

    # After smoothing, the noise will decrease by ~1/beam_pixels where
    # beam_pixels = (beam_fwhm / pixel_size)^2. So we add extra noise now and
    # then smooth it out later
    noise_factor = (beam_fwhm / pixel_size).to("").value ** 2.0
    TB += np.random.randn(*TB.shape) * noise * noise_factor

    # Convolve with beam
    beam_sigma = beam_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    beam_sigma_pix = (beam_sigma / pixel_size).to("").value
    TB_smooth = spatial_smooth(TB, beam_sigma_pix)

    # Saving to a fits file
    hdu = fits.PrimaryHDU(TB_smooth.to(u.K).value.T)
    hdu.header = hdr
    hdu.header["OBJECT"] = "Synthetic HII Region Observation"
    hdu.header["BMAJ"] = beam_fwhm.to("deg").value
    hdu.header["BMIN"] = beam_fwhm.to("deg").value
    hdu.writeto(f"fits/{filename}.fits", overwrite=True)
    pass

class HIIRegion:
    """
    The attributes and physics of a homogeneous, isothermal, spherical HII region.
    """

    def __init__(
        self,
        radius=1.0 * u.pc,
        distance=1.0 * u.kpc,
        electron_temperature=1e4 * u.K,
        electron_density=1000 * u.cm**-3,
        velocity=0 * u.km/u.s,
    ):
        """
        Initialize a new HIIRegion object.

        Inputs:
            radius -- Radius (with units)
            distance -- Distance (with units)
            temperature -- Electron temperature (with units)
            density -- Electron density (with units)

        Returns:
            hii -- New HIIRegion instance
        """
        self.radius = radius
        self.distance = distance
        self.electron_temperature = electron_temperature
        self.electron_density = electron_density
        self.velocity = velocity


class Simulation:
    """
    The parameters of a synthetic simulation of a simulated HII region.
    """

    def __init__(
        self,
        pixel_size=10.0 * u.arcsec,
        npix=300,
        lospix=300,
        nchan=200,
        channel_size=20.0 * u.kHz,
        rrl_freq=6.0 * u.GHz,
    ):
        """
        Initialize a new Observation object.

        Inputs:
            pixel_size -- Width of pixel in synethetic data cube (with units)
            npix -- Width of the sythetic data cube (in pixels)
            nchan -- Number of frequency channels
            channel_size -- Width of frequency channel (with units)
            rrl_freq -- Rest frequency of RRL transition (with units)

        Returns:
            obs -- New Observation instance
        """
        self.pixel_size = pixel_size
        self.npix = npix
        self.lospix = lospix
        self.imsize = self.pixel_size * self.npix
        self.nchan = nchan
        self.channel_size = channel_size
        self.rrl_freq = rrl_freq

        # define frequency and velocity axes
        freq_width = self.nchan * self.channel_size
        self.freq_axis = np.linspace(
            self.rrl_freq - freq_width / 2.0,
            self.rrl_freq + freq_width / 2.0,
            self.nchan,
            endpoint=True,
        )
        self.velo_axis = doppler(self.rrl_freq, self.freq_axis)

    def simulate(self, hiiregion, filename):
        """
        Generate a synthetic radio continuum + radio recombination line simulation
        of an HII region, and save the resulting data cube to a FITS file. The RRL
        is placed at the center of the spectral axis. This doesn't account for
        noise or beam width.

        Inputs:
            hiiregion -- HIIRegion object to be "observed"
            filename -- Output phys FITS filename

        Returns:
            Nothing
        """
        # image size in physical units
        imsize_physical = self.imsize.to("rad").value * hiiregion.distance

        #calculating thermal broadening
        rrl_fwhm_freq = thermal_fwhm(hiiregion.electron_temperature, self.rrl_freq)

        # Finding where sphere is defined
        spheremap = generate_bool_sphere(hiiregion.radius, self.npix, imsize_physical,self.lospix)

        # Truncating off density outside sphere
        sphere_density = hiiregion.electron_density * spheremap

        # Calculating the emission measure for each voxel
        vox_depth = imsize_physical/self.lospix
        em_grid = emission_measure(sphere_density,vox_depth)

        # Using the input velocites as gaussian line centers
        dop_rrl_freq = doppler_freq(self.rrl_freq,hiiregion.velocity)
            
        # Stacking and adding together channels
        tau_rrl = np.zeros((self.npix,self.npix,self.nchan))
        print("Solving each channel of simulation...")
        for channel in tqdm(range(self.nchan)):
            single_channel_3d = rrl_opacity(
                temp = hiiregion.electron_temperature,
                em = em_grid,
                nu = self.freq_axis[channel],
                center_freq = dop_rrl_freq,
                fwhm_freq = rrl_fwhm_freq,
            )
            single_channel = np.sum(single_channel_3d,axis=2)
            tau_rrl[:,:,channel] = single_channel

        # Free-free opacity
        tau_ff = ff_opacity(hiiregion.electron_temperature, self.freq_axis, np.sum(em_grid,axis=2))

        # Brightness temperature
        taus = [tau_ff, tau_rrl, tau_ff+tau_rrl]
        taunames = ["ff","rrl","both"]
        for tau, tauname in zip(taus,taunames):
            TB = brightness_temp(tau, hiiregion.electron_temperature)

            # Saving to a fits file
            hdu = fits.PrimaryHDU(TB.to(u.K).value.T)
            hdu.header["OBJECT"] = "Synthetic HII Region Simulation"
            hdu.header["CRVAL1"] = 0.0
            hdu.header["CRVAL2"] = 0.0
            hdu.header["CRVAL3"] = 0.0
            hdu.header["CTYPE3"] = "VELO-LSR"
            hdu.header["CTYPE1"] = "RA---TAN"
            hdu.header["CTYPE2"] = "DEC--TAN"
            hdu.header["CRPIX1"] = TB.shape[0] / 2 + 0.5
            hdu.header["CRPIX2"] = TB.shape[1] / 2 + 0.5
            hdu.header["CRPIX3"] = TB.shape[2] / 2 + 0.5
            hdu.header["CDELT3"] = (self.velo_axis[1] - self.velo_axis[0]).to("km/s").value
            hdu.header["CDELT1"] = self.pixel_size.to("deg").value
            hdu.header["CDELT2"] = self.pixel_size.to("deg").value
            hdu.header["BTYPE"] = "Brightness Temperature"
            hdu.header["BUNIT"] = "K"
            hdu.header["CUNIT1"] = "deg"
            hdu.header["CUNIT2"] = "deg"
            hdu.header["CUNIT3"] = "km/s"
            hdu.header["BPA"] = 0.0
            hdu.header["RESTFRQ"] = self.rrl_freq.to(u.Hz).value
            hdu.writeto(f"sim/{filename+tauname}sim.fits", overwrite=True)
            pass

def split_observations(filenamebase,beam_fwhm,noise):
    """
    Generates observation for each of the simulated regions, between RRLs, FF, and combined.

    Inputs:
    filenamebase -- String used to generate simulation files
    beam_fwhm -- Beam full-width half max
    noise -- Noise to be applied

    Outputs:
    Nothing
    """
    tempnames = ["ff","rrl","both"]
    for temp in tempnames:
        observe(f"{filenamebase+temp}sim.fits",
                f"{filenamebase+temp}_{beam_fwhm.value}",
                beam_fwhm=beam_fwhm,
                noise=noise,
                )

    pass

def main():
    # Synthetic observation
    impix = 300
    dens1, vel1 = gen_turbulence(impix,
                                 mean_density=1000*u.cm**-3,
                                 seed=100,
                                 mach_number=5,
                                 driving_parameter=0.75,
                                 )

    region1 = HIIRegion(
        electron_density = dens1,
        velocity = vel1,
        distance=0.25*u.kpc,
    )
    obs1 = Simulation(
        nchan=200,
        npix=dens1.shape[0],
        lospix = dens1.shape[2],
        pixel_size=50*u.arcsec/(impix/50)/(region1.distance/0.25/u.kpc),
        channel_size=40*u.kHz,
    )
    obs1.simulate(region1, "region1")

    split_observations("region1", beam_fwhm=200*u.arcsec, noise=0.01*u.K)

if __name__ == "__main__":
    main()
