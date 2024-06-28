# Eliza Canales -- Summer 2024
# The ultimate purpose of this code is to generate a fits
# file of an HII region with turbulence applied

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
from scipy.ndimage import gaussian_filter

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

def sphere_depth_gen(radius,imsize,impix):
    '''
    radius -- Radius of the sphere being used, can be visual
    radius or physical radius
    imsize -- Size of the image, must be same unit as radius
    impix  -- The number of pixels in the x and y directions

    This function takes the three listed constants and returns
    a matrix of the distance from to the front to back at each
    of the pixels, using the center of the pixel to calculate it.

    The distance calculated is based off if the object is a
    perfect sphere.
    '''
    #Delta between each pixel
    pixel_dist = imsize / impix

    midpoint = imsize / 2 #Where there is a depth of the diameter
    stepper = np.linspace(0,imsize.value,impix) * imsize.unit + pixel_dist/2
    i, j = np.meshgrid(stepper,stepper,indexing='ij')
    sqrdepth = radius**2 - (midpoint-i)**2 - (midpoint-j)**2
    depth = 2*np.sqrt(sqrdepth * (sqrdepth>0))
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
    return gaussian_filter(data_cube,[radius,radius,0])

class HIIRegion:
    '''
    Purpose of structure is to be a fully homogenous,
    isothermal, spherical HII region. 

    When initialized, a radius, temperature, density,
    and distance from Earth must be passed to it.
    '''
    def __init__(self, radius=1*u.parsec, temperature=1e4*u.K, density=1000*u.cm**-3):
        '''
        radius -- Radius of the HII region, astropy length
        quantity.
        temperature -- Temperature of the HII region as an
        astropy temperature quantity.
        density -- Number density of the HII region, may be
        an astropy number density quantity or an array of 
        quantities.
        '''
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
        '''
        pixwid   -- Width of pixel in astropy angle quantity.
        imwid    -- Number of pixels in each dimension of image.
        beamsize -- Beam width as astropy angle quantity.
        noise    -- Amount of noise to be applied as astropy
        temperature quantity.
        channels -- Number of channels to be generated in data
        cube for observation.
        rrl_freq -- Center frequency that telescope is observing,
        also where the peak of the RRL emission, as astropy
        frequency quantity.
        '''
        self.pixwid = pixwid
        self.imwid = imwid
        self.beamsize = beamsize
        self.noise = noise
        self.channels = channels
        self.rrl_freq = rrl_freq

    def observe(self,hiiregion,filename):
        '''
        hiiregion -- An HII region to be observed by the telescope
        filename  -- Filename of fits file to be made, images will
        save in fits/ with a .fits extension.

        The observe function takes in an HII region and produces a
        mock observation of it based on the attributes of the 
        telescope and the region. This function assumes the 
        existence of an RRL at the telescope's center frequency.

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

        # Applying the RRLs from equation 7.97 (math with em moved to after distribution made)
        rrl_temp_peak = 1.92e3 * (hiiregion.temperature/u.K)**(-3/2)  * (rrl_fwhm/u.kHz)**-1 * u.K
        rrl_temp = rrl_temp_peak * np.exp(-1/2 * (2.35 * (self.rrl_freq-nu) / rrl_fwhm)**2)
        tempbright += rrl_temp * (em[...,None] / (u.pc * u.cm**-6))

        # Adding noise and smoothing out
        tempbright += np.random.normal(size=tempbright.shape) * self.noise
        observation = blurring_agent(tempbright,self.beamsize/self.pixwid * (8*np.log(2))**(-1/2)) * tempbright.unit #Extra factor in smoothing because beam size is the FWHM of blurring
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
    testregion = HIIRegion()

    # Model nebula powered by type O6 star
    nebO6 = HIIRegion(1.25*u.pc, 4e3*u.K, 200*u.cm**-3)

    # Creating a test telescope
    testtelescope = Telescope()
    testtelescope.observe(testregion,'testfits')
    testtelescope.observe(nebO6,'nebulaO6')

if __name__ == "__main__":
    main()

