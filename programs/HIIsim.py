import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

'''
Future plans for this program:

In the HII Region class for RRLs:
-Add transition frequencies for each transition
-Add absorption rates for each transition 
-Add spontaneous emission rates for each transition

When these are added, computationally find the equibrium
densities and transition rates for each energy level
This can be used to estimate the intensity of each 
transition before thermal broadening is applied

For the continuum:
-Understand 
'''

class HIIRegion:
    '''
    Purpose of structure is to be a fully homogenous,
    isothermal, spherical HII region. This also contains
    the electron density and temperature.

    When initialized, a radius, temperature, and density
    must be passed to it.
    '''
    def __init__(self, radius, temperature, density):
        self.radius = radius
        self.temperature = temperature
        self.density = density

if __name__ == "__main__":
    main()

