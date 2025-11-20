"""
getproperties.py
Records properties of HII regions from RRL images and their moment maps    
Eliza Canales
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.io import fits
import argparse
from turbustat.statistics import PowerSpectrum
import matplotlib.pyplot as plt

def plotStats(array, title,colormap):

    plt.imshow(array,origin='lower',extent=(0.25,5.25,0,600),aspect='auto')
    plt.title(title)
    plt.xlabel("Mach number times driving number")
    plt.ylabel("Resolution (arcsecs)")
    plt.colorbar()
    plt.show()

def main(filebase):

    seeds=['1','2','3','4','5']
    mds=['1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
    reses=['50.0','150.0','250.0','350.0','450.0','550.0']

    #Prepare histogram
    pspecslope1=np.zeros((len(reses),len(mds)))
    allmeanM2=np.zeros((len(reses),len(mds),len(seeds)))
    mediandM2=np.zeros((len(reses),len(mds)))

    #Load power spectrum results
    for s in range(len(seeds)):
        for m in range(len(mds)):
            for r in range(len(reses)):
                pspec1 = PowerSpectrum.load_results(filebase+'_s'+seeds[s]+'_md'+mds[m]+'rrl_'+reses[r]+'_M1.pkl')
                #pspec2 = PowerSpectrum.load_results(filebase+'_s'+seed+'_md'+md+'rrl_'+res+'M2.pkl')
                
                pspecslope1[r][m] += pspec1.slope / len(seeds)
                hdul = fits.open(filebase+'_s'+seeds[s]+'_md'+mds[m]+'rrl_'+reses[r]+'_M2.fits')
                allmeanM2[r][m][s] = np.nanmean(hdul[0].data)
                hdul.close()

                hdul = fits.open(filebase+'_s'+seeds[s]+'_md'+mds[m]+'rrl_'+reses[r]+'_dM2.fits')
                mediandM2[r][m] += np.nanmedian(hdul[0].data) / len(seeds)
                hdul.close()

                
    meanM2 = np.mean(allmeanM2,axis=2)
    stdmeanM2 = np.std(allmeanM2,axis=2)
    #Plot it
    hdul = fits.open(filebase+'_s'+seeds[s]+'_md'+mds[m]+'rrl_'+reses[r]+'_M2.fits')
    pspec1 = PowerSpectrum.load_results(filebase+'_s'+seeds[0]+'_md'+mds[int(len(mds)/2)]+'rrl_'+reses[int(len(reses)/2)]+'_M1.pkl')
    pspec1.plot_fit()
    plotStats(pspecslope1,"Average 1D Power Slope","vicidis")
    plotStats(meanM2,"Average of Moment 2","viridis")
    plotStats(mediandM2,"Median Uncertainty of Moment 2")
    plotStats(stdmeanM2,"Standard Deviation of Moment 2 Seeds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates properties of HII images and moment maps",
        prog="figuremaking.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filebase",
        type=str,
        default="region1",
        help="Filename",
    )
    args = parser.parse_args()
    main(**vars(args))
