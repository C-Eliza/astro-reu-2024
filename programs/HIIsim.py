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
    print(raw_phi)
    phi = c.c / nu_0 * (c.m_p / (2 * np.pi * c.k_B * temp))**(1/2) * raw_phi
    return(phi.to(u.ns))

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
    nu = np.linspace(0.99*nu_0,1.01*nu_0)
    phi = line_broadening(temp,nu_0,nu)
    print(phi)
    #plt.plot(nu,phi)
    #plt.show()

if __name__ == "__main__":
    main()

