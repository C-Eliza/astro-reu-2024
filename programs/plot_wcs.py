import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
import argparse 

def main(fname,cubemode):
    hdu = fits.open(fname)[0]
    if cubemode:
        data = hdu.data[int(hdu.data.shape[0]/2)]
        xsize = hdu.data.shape[1]
        ysize = hdu.data.shape[2]
    else:
        data = hdu.data
        xsize = hdu.data.shape[0]
        ysize = hdu.data.shape[1]

    # limit color range
    vmin = np.nanpercentile(data, 5.0)
    vmax = np.nanpercentile(data, 95.0)

    # generate world coordinate system
    wcs = WCS(hdu.header)

    # plot data
    fig = plt.figure()
    ax = plt.subplot(projection=wcs.celestial)
    if "M1" in fname or "vlsr" in fname:
        colormap = "cividis"
    elif "M2" in fname or "fwhm" in fname:
        colormap = "viridis"
    else:
        colormap = "magma"

    cax = ax.imshow(
        np.abs(data) if cubemode else data,
        origin='lower',
        interpolation='none',
        cmap=colormap,
        norm = LogNorm(vmin=0.1,vmax=vmax) if cubemode else Normalize(vmin=vmin,vmax=vmax)
    )
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Declination (J2000)")

    # zoom in
    xcenter = xsize // 2
    ycenter = ysize // 2
    xmin = xcenter - xsize // 4
    ymin = ycenter - ysize // 4
    xmax = xcenter + xsize // 4
    ymax = ycenter + ysize // 4
    ax.set_xlim(0, xsize)
    ax.set_ylim(0, ysize)

    # plot beam
    pixsize = hdu.header['CDELT2']
    beam_maj = hdu.header["BMAJ"] / pixsize
    beam_min = hdu.header["BMIN"] / pixsize
    beam_pa = hdu.header["BPA"]
    beam_scale = beam_maj / xsize
    ellipse = Ellipse(
        ((0.1+beam_scale)*xcenter, (0.1+beam_scale)*ycenter),
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
    if "M1" in fname or "vlsr" in fname: 
        cbar.set_label(r"$V_{\rm LSR}$ (km s$^{-1}$)")
    elif "M2" in fname or "fwhm" in fname:
        cbar.set_label(r"$\Delta V$ (km s$^{-1}$)")
    else:
        cbar.set_label(r"$T_b$ (Absolute K)")

    fname = fname.replace('.fits', '.png')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Makes pictures of moment maps",
        prog="plot_wcs.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fname",
        type=str,
        default="region1",
        help="Filename",
    )
    parser.add_argument(
        "--cubemode",
        action="store_true",
        help="Is this a datacube?",
    )
    args = parser.parse_args()
    main(**vars(args))
