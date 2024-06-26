# Eliza Canales -- Summer 2024
# The ultimate purpose of this code is to generate a fits
# file of an HII region with turbulence applied

import numpy as np
import astropy.units as u
import astropy.constants as c
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

def blurring_agent(flux_density,radius):
    '''
    flux_density -- Data cube of true flux density values
    radius       -- The radius (int) of the smoothing for the
    image to use to simulate a radio observation

    This function simply cause spatial smoothing in each frame
    '''
    return ndi.gaussian_filter(flux_density,[radius,radius,0])

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
    def __init__(self, pixwid, imwid, beamsize, noise):
        self.pixwid = pixwid
        self.imwid = imwid
        self.beamsize = beamsize
        self.noise = noise

    def observe(self,hiiregion,rrl_freq,samples):
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
        rrl_fwhm = thermal_fwhm(hiiregion.temperature,rrl_freq)
        nu = np.linspace((rrl_freq-3*rrl_fwhm).to(u.GHz),(rrl_freq+3*rrl_fwhm).to(u.GHz),samples)

        # Generating the emission measure for the region
        depthmap = sphere_depth_gen(hiiregion.radius,4*hiiregion.radius,self.imwid)
        em = emission_measure(hiiregion.density,depthmap)

        # Creating the image before any noise or smoothing
        tau = ff_opacity(hiiregion.temperature,nu,em)
        tempbright = temp_brightness(tau,hiiregion.temperature)

        # Applying the RRLs from equation 7.97
        rrl_temp_raw = 1.92e3 * (hiiregion.temperature/u.K)**(-3/2) * (em[...,None]/(u.pc*u.cm**-6)) * (rrl_fwhm/u.kHz)**-1 * u.K
        rrl_temp = rrl_temp_raw[...,None] * np.exp(-1/2 * (2.35 * (rrl_freq-nu) / rrl_fwhm)**2)
        tempbright += np.sum(rrl_temp, axis=2)

        # Adding noise and smoothing out
        tempbright += np.random.normal(size=tempbright.shape) * self.noise
        observation = blurring_agent(tempbright,self.beamsize/self.pixwid) * tempbright.unit
        velocities = freq_to_velocity(rrl_freq, nu)
        return observation, velocities

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
    testtelescope = Telescope(pixwid,imwid,beamsize,10*u.K)

    # Generating a test observation
    nu_0 = 4 * u.GHz
    observation, velocities = testtelescope.observe(testregion,nu_0,200)

    '''
    # Plotting the thermal broadening
    plt.plot(velocities,phi)
    plt.title('Thermal broadening at 10000K in 6GHz')
    plt.xlabel("Doppler shift (km/s)")
    plt.ylabel("Distribution (ns?)")
    '''
    '''
    # Showing test observations
    fig = plt.figure()
    fig.suptitle('Test observation of HII region')

    ax1 = fig.add_subplot(121)
    ax1.set_title('4 GHz')
    ax1.xaxis.set_major_locator(ticker.NullLocator())
    ax1.yaxis.set_major_locator(ticker.NullLocator())
    im = ax1.imshow(observation[:,:,0].value, interpolation='nearest',vmax=np.max(observation.value))

    ax2 = fig.add_subplot(122)
    ax2.set_title('6 GHz')
    ax2.xaxis.set_major_locator(ticker.NullLocator())
    ax2.yaxis.set_major_locator(ticker.NullLocator())
    im = ax2.imshow(observation[:,:,3].value, interpolation='nearest',vmax=np.max(observation.value))

    fig.canvas.draw()
    plt.colorbar(im, ax=[ax1,ax2],fraction=0.046, pad=0.04,shrink=0.8)
    plt.show()
    '''
    # Generate a basic spectra
    plt.plot(velocities.to(u.km/u.s),np.average(observation.to(u.K)*(observation>np.max(observation)/10),axis=(0,1)))
    plt.title("RRL at approximately 4 GHz")
    plt.xlabel("Doppler shift (km/s)")
    plt.ylabel("Brightness temperature (K)")
    plt.show()
    '''
    plt.imshow(observation[:,:,0].to(u.K).value)
    plt.show()
    '''

if __name__ == "__main__":
    main()

