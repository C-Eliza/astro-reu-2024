from astropy.io import fits
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import bettermoments as bm

def makespectra(filename,title):
    data, velax = bm.load_cube(filename)
    plt.figure(figsize=(8,3),dpi=80)
    plt.title(title)
    plt.plot(velax, np.mean(data, axis=(1,2)))
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Temperature Brightness (K)")
    plt.grid()
    plt.show()

makespectra("sim/noturbffsim.fits","Free-free spectrum")
makespectra("sim/noturbrrlsim.fits","Radio recombination line spectrum")
