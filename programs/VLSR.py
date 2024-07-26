import argparse
import numpy as np
from astropy.io import fits
import bettermoments as bm
import matplotlib.pyplot as plt


def gen_VLSR(path):
    data, velax = bm.load_cube(path)
    smoothed_data = bm.smooth_data(data=data, smooth=3, polyorder=0)
    rms = bm.estimate_RMS(data=data, N=50)

    # rms = 0.01
    threshold_mask = bm.get_threshold_mask(
        data=smoothed_data, clip=6.0, smooth_threshold_mask=0.0
    )
    masked_data = smoothed_data * threshold_mask
    moments = bm.collapse_first(velax=velax, data=masked_data, rms=rms)
    bm.save_to_FITS(moments=moments, method="first", path=path)


def get_velocity_data(path):
    moments = fits.open(path)[0].data
    return moments


def main(fnames):
    impix = [int(fname.split("_")[1].replace("rrl", "")) for fname in fnames]
    velocity_data = []

    for fname in fnames:
        gen_VLSR(fname)
        M1_fname = fname.replace(".fits", "_M1.fits")
        velocity_data.append(get_velocity_data(M1_fname))

    # plot 5, 95 percentile difference vs. impix
    vel_diff = [np.diff(np.nanpercentile(data, [5.0, 95.0])) for data in velocity_data]
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(impix, vel_diff, "ko")
    ax.set_xlabel("impix")
    ax.set_ylabel(r"$V_{95\%} - V_{5\%}$ (km s$^{-1}$)")
    fig.savefig("figures/vel_diff_percentile.png")
    plt.close(fig)

    # plot histograms
    fig, ax = plt.subplots(layout="constrained")
    bins = np.linspace(-50.0, 50.0, 50)
    for label, data in zip(impix, velocity_data):
        ax.hist(data.flatten(), bins=bins, label=label, density=True, histtype="step")
    ax.legend(loc="best")
    ax.set_xlabel(r"V (km s$^{-1}$)")
    ax.set_ylabel("Number")
    fig.savefig("figures/vel_histogram.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate velocity statistics",
        prog="VLSR.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fnames",
        type=str,
        nargs="+",
        help="RRL data cubes",
    )
    args = parser.parse_args()
    main(**vars(args))
