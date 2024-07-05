import numpy as np
from astropy.io import fits
import bettermoments as bm

path = 'fits/testfits3d_200.fits'
data, velax = bm.load_cube(path)
smoothed_data = bm.smooth_data(data=data, smooth=3, polyorder=0)
rms = bm.estimate_RMS(data=smoothed_data, N=5)
threshold_mask = bm.get_threshold_mask(data=smoothed_data,
                                       clip=2.0,
                                       smooth_threshold_mask=0.0)
masked_data = smoothed_data * threshold_mask
moments = bm.collapse_first(velax=velax, data=masked_data, rms=rms)
bm.save_to_FITS(moments=moments, method='first', path=path)
