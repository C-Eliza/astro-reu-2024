import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main(fname):
    hdu = fits.open(fname)[0]

    # snip out weird values
    print(hdu.data)
    rms = np.nanstd(hdu.data)
    average = np.nanmedian(hdu.data)
    hdu.data = hdu.data + np.where(rms/5>np.abs(hdu.data), 0, np.nan)
    print(hdu.data[0,0])
    
    # limit color range
    vmin = np.nanpercentile(hdu.data, 1.0)
    vmax = np.nanpercentile(hdu.data, 99.0)

    # generate world coordinate system
    wcs = WCS(hdu.header)

    # plot data
    fig = plt.figure()
    ax = plt.subplot(projection=wcs.celestial)
    cax = ax.imshow(
        hdu.data,
        origin='lower',
        interpolation='none',
        cmap='inferno',
        vmin=vmin,
        vmax=vmax
    )
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Declination (J2000)")

    # zoom in
    xsize = hdu.data.shape[0]
    ysize = hdu.data.shape[1]
    xcenter = xsize // 2
    ycenter = ysize // 2
    xmin = xcenter - xsize // 2
    ymin = ycenter - ysize // 2
    xmax = xcenter + xsize // 2
    ymax = ycenter + ysize // 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # plot beam
    pixsize = hdu.header['CDELT2']
    beam_maj = hdu.header["BMAJ"] / pixsize
    beam_min = hdu.header["BMIN"] / pixsize
    beam_pa = hdu.header["BPA"]
    ellipse = Ellipse(
        (0.3*xcenter, 0.3*ycenter),
        beam_min,
        beam_maj,
        angle=beam_pa,
        fill=True,
        zorder=10,
        hatch='///',
        edgecolor='black',
        facecolor='white'
    )
    ax.add_patch(ellipse)

    # add colorbar
    # N.B. the values for "fraction" and "pad" are "magic numbers":
    # they "always work" in my experience
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$V_{\rm LSR}$ (km s$^{-1}$)")

    fname = fname.replace('.fits', '.pdf')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    main("fits/testfits3d_200_M1.fits")
    main("fits/testfits3d_800_M1.fits")
