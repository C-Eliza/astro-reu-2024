import numpy as np
from astropy.io import fits
import bettermoments as bm

def gen_VLSR(path):
    data, velax = bm.load_cube(path)
    smoothed_data = bm.smooth_data(data=data, smooth=3, polyorder=0)
    rms = bm.estimate_RMS(data=smoothed_data, N=5)
    threshold_mask = bm.get_threshold_mask(data=smoothed_data,
                                           clip=15.0,
                                           smooth_threshold_mask=0.0)
    masked_data = smoothed_data * threshold_mask
    moments = bm.collapse_first(velax=velax, data=masked_data, rms=rms)
    bm.save_to_FITS(moments=moments, method='first', path=path)

def velocity_range(path):
    moments = fits.open(path)[0].data
    low = np.nanpercentile(moments,5)
    high = np.nanpercentile(moments,95)
    return(low,high)

gen_VLSR("fits/region1rrl_200.0.fits")
print(velocity_range("fits/region1rrl_200.0_M1.fits"))
