# Eliza Canales -- Summer 2024
# The ultimate purpose of this code is to generate a fits
# file of an HII region with turbulence applied

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.ndimage as ndi

def thermal_fwhm(temp, nu_0):
    '''
    temp -- Temperature of the hydrogen
    nu_0 -- Transition frequency

    Returns FWHM of thermal broadening
    '''
    return np.sqrt(8*np.log(2)*c.k_B/c.c**2) * np.sqrt(temp/c.m_p) * nu_0

def freq_to_velocity(nu_0,nu):
    '''
    Takes in the laboratory frame frequency of emission and
    returns the doppler shift in kilometers per second.
    '''
    return c.c * (nu-nu_0) / nu_0

def emission_measure(density,depth):
    '''
    density -- Average electron density of an HII region
    along a path
    depth   -- Distance from front to back of an HII region,
    assuming constant density

    The depth given should be a matrix, as the emission measure
    returned should be a matrix

    Takes in the average density of an HII region and the 
    total distance from front to back of an observed point
    on the HII region.

    This function uses equation 4.57 from the NRAO web textbook

    The emission measure is then returned
    '''
    return density**2 * depth

def ff_opacity(temp,nu,em):
    '''
    temp -- Temperature of the region, typically in Kelvin
    nu   -- Frequency which the opacity is being calculated for
    em   -- Emission measure of the position being measured

    The incoming nu is a range of frequencies, em is a matrix,
    and a 'data cube' of free-free opacity is returned

    This function uses equation 4.60 from the NRAO web textbook

    This function returns the free-free opacity of a region with
    a given temperature and emission measure for a frequency.
    '''
    return 3.28e-7 * (temp/(1e4*u.K))**-1.35 * (nu/u.GHz)**-2.1 * (em[...,None]/(u.pc*u.cm**-6))

def sphere_depth_gen(radius,length,steps):
    '''
    radius -- Radius of the sphere being used, can be visual
    radius or physical radius
    length -- Size of the image, must be same unit as radius
    steps  -- The number of pixels in the x and y directions

    This function takes the three listed constants and returns
    a matrix of the distance from to the front to back at each
    of the pixels, using the center of the pixel to calculate it.

    The distance calculated is based off if the object is a
    perfect sphere.
    '''
    #Delta between each pixel
    pixel_dist = length / steps

    midpoint = length / 2 #Where there is a depth of the diameter
    stepper = np.array([np.linspace(0,length,steps) + pixel_dist/2]) * length.unit
    depth = np.nan_to_num(2*np.sqrt(radius**2 - (midpoint-stepper)**2 - (midpoint-stepper.T)**2))
    return depth

def temp_brightness(depth,temp):
    '''
    depth -- The optical depth 'data cube' to have a brightness
    temperature found for
    temp  -- Electron temperature

    Returns the brightness temperature for a given optical depth
    and electron temp
    '''
    return temp * (1-np.exp(-depth))

def flux_density(intensity,delta):
    '''
    intensity -- Intensity 'data cube'
    delta     -- Width of each pixel in arcseconds
    
    Calculates flux density from intensity via equation 2.10
    '''
    return intensity * delta**2 / u.rad**2

def blurring_agent(data_cube,radius):
    '''
    data_cube -- Data cube to be smoothed over
    radius    -- The radius (int) of the smoothing for the
    image to use to simulate a radio observation

    This function simply cause spatial smoothing in each frame
    '''
    return ndi.gaussian_filter(data_cube,[radius,radius,0])

class HIIRegion:
    '''
    Purpose of structure is to be a fully homogenous,
    isothermal, spherical HII region. 

    When initialized, a radius, temperature, density,
    and distance from Earth must be passed to it.
    '''
    def __init__(self, radius, temperature, density):
        self.radius = radius
        self.temperature = temperature
        self.density = density

class Telescope:
    '''
    Purpose is to hold all the information about a telescope
    that one needs in order to generate a fake observation.

    When initialized, it takes in a pixel width in arcseconds
    and the width of an image it generates in pixels. It also
    takes in a beam area (pix) to simulate gaussian effects.
    '''
    def __init__(self, pixwid = 1*u.arcsec, imwid = 100, beamsize = 10*u.arcsec, noise = 1*u.K, channels = 200, rrl_freq = 6*u.GHz):
        self.pixwid = pixwid
        self.imwid = imwid
        self.beamsize = beamsize
        self.noise = noise
        self.channels = channels
        self.rrl_freq = rrl_freq

    def observe(self,hiiregion,filename):
        '''
        hiiregion -- An HII region to be observed by the telescope
        nu        -- A collection of frequencies to have images
        produced for

        The observe function takes in an HII region and produces a
        mock observation of it based on the attributes of the 
        telescope and the region. A range of frequencies must also
        be provided to create the observation.

        This accounts for continuum as well as RRLs.
        '''
        #Creating frequencies to check over
        rrl_fwhm = thermal_fwhm(hiiregion.temperature,self.rrl_freq)
        nu = np.linspace((self.rrl_freq-3*rrl_fwhm).to(u.GHz),(self.rrl_freq+3*rrl_fwhm).to(u.GHz),self.channels)

        # Generating the emission measure for the region
        depthmap = sphere_depth_gen(hiiregion.radius,4*hiiregion.radius,self.imwid)
        em = emission_measure(hiiregion.density,depthmap)

        # Creating the image before any noise or smoothing
        tau = ff_opacity(hiiregion.temperature,nu,em)
        tempbright = temp_brightness(tau,hiiregion.temperature)

        # Applying the RRLs from equation 7.97
        rrl_temp_raw = 1.92e3 * (hiiregion.temperature/u.K)**(-3/2) * (em[...,None]/(u.pc*u.cm**-6)) * (rrl_fwhm/u.kHz)**-1 * u.K
        rrl_temp = rrl_temp_raw[...,None] * np.exp(-1/2 * (2.35 * (self.rrl_freq-nu) / rrl_fwhm)**2)
        tempbright += np.sum(rrl_temp, axis=2)

        # Adding noise and smoothing out
        tempbright += np.random.normal(size=tempbright.shape) * self.noise
        observation = blurring_agent(tempbright,self.beamsize/self.pixwid) * tempbright.unit
        velocities = freq_to_velocity(self.rrl_freq, nu)

        # Saving to a fits file
        hdu = fits.PrimaryHDU(np.transpose(observation.to(u.K).value,(2,0,1)))
        hdu.header['OBJECT'] = 'Sample RRL Observation'
        hdu.header['CRVAL1'] = 0.0
        hdu.header['CRVAL2'] = 0.0
        hdu.header['CRVAL3'] = 0.0
        hdu.header['CTYPE3'] = 'VEL'
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRPIX1'] = observation.shape[0]/2 + 1/2
        hdu.header['CRPIX2'] = observation.shape[1]/2 + 1/2
        hdu.header['CRPIX3'] = observation.shape[2]/2 + 1/2
        hdu.header['NAXIS1'] = observation.shape[0]
        hdu.header['NAXIS2'] = observation.shape[1]
        hdu.header['NAXIS3'] = observation.shape[2]
        hdu.header['CDELT3'] = (velocities[1] - velocities[0]).to(u.km/u.s).value
        hdu.header['CDELT1'] = (self.pixwid / u.deg).to(u.m/u.m).value
        hdu.header['CDELT2'] = (self.pixwid / u.deg).to(u.m/u.m).value
        hdu.header['BTYPE'] = 'Brightness Temperature'
        hdu.header['BUNIT'] = 'K'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CUNIT3'] = 'km/s'
        hdu.header['BMAJ'] = (self.beamsize / u.deg).to(u.m/u.m).value
        hdu.header['BMIN'] = (self.beamsize / u.deg).to(u.m/u.m).value
        hdu.header['BPA'] = 0.0
        hdu.header['RESTFRQ'] = self.rrl_freq.to(u.Hz).value
        hdu.writeto('fits/'+filename+'.fits',overwrite=True)
        pass

def main():
    # Creating a test region
    radius = 1 * u.pc
    temp = 1e4 * u.K
    n_e = 1000 * u.cm**-3
    dist = 1e4 * u.pc
    testregion = HIIRegion(radius, temp, n_e)

    # Model nebula powered by type O6 star
    nebO6 = HIIRegion(1.25*u.pc, 4e3*u.K, 200*u.cm**-3)

    # Creating a test telescope
    pixwid = 1 * u.arcsec
    imwid = 100
    beamsize = 5 * u.arcsec
    testtelescope = Telescope()
    testtelescope.observe(testregion,'testfits')

if __name__ == "__main__":
    main()

