import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

'''
Future plans for this program:

In the HII Region class for RRLs:
-Add transition frequencies for each transition
-Add absorption rates for each transition 
--Use equation 7.32 from NRAO for gaussian
-Add spontaneous emission rates for each transition

When these are added, computationally find the equibrium
densities and transition rates for each energy level

For the continuum:
-Understand 
'''

def line_broadening(temp, nu_0, nu):
    '''
    Takes the temperature of the HII(temp), the transition 
    frequency (nu_0), and a range of frequencies to test (nu).

    Returns the gaussian value for each provided frequency
    '''
    # Calculation broken into two parts due to complexity
    # raw_phi creates the gaussian profile without the constant
    # in front
    raw_phi = np.exp( -(c.m_p * c.c**2) / (2 * c.k_B * temp) * (nu - nu_0)**2 / nu_0**2 )
    phi = c.c / nu_0 * (c.m_p / (2 * np.pi * c.k_B * temp))**(1/2) * raw_phi
    return(phi.to(u.ns))

def freq_to_velocity(nu_0,nu):
    '''
    Takes in the laboratory frame frequency of emission and
    returns the doppler shift in kilometers per second.
    '''
    return (c.c * (nu-nu_0) / nu_0).to(u.km/u.s)

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
    return((density**2*depth).to(u.pc*u.cm**-6))

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

    # Moves frequency to the third axis so the cube is created
    nu = nu[None,None,:]

    return((3.28e-7 * (temp/(1e4*u.K))**-1.35 * (nu/u.GHz)**-2.1 * (em/(u.pc*u.cm**-6))).to(u.dimensionless_unscaled))

class HIIRegion:
    '''
    Purpose of structure is to be a fully homogenous,
    isothermal, spherical HII region. 

    When initialized, a radius, temperature, and density
    must be passed to it.
    '''
    def __init__(self, radius, temperature, density):
        self.radius = radius
        self.temperature = temperature
        self.density = density

def main():
    temp = 80 * u.K
    nu_0 = 6 * u.GHz
    nu = np.linspace(0.99999*nu_0,1.00001*nu_0)
    phi = line_broadening(temp,nu_0,nu)
    velocities = freq_to_velocity(nu_0,nu)
    print(sum(phi*(nu[1]-nu[0])).to(u.s/u.s))
    #print(phi)
    #plt.plot(nu,phi)
    #plt.show()

if __name__ == "__main__":
    main()

