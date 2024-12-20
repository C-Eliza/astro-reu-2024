"""
fit_gradient.py

Fit a plane to a FITS image.

Original: Trey Wenger - June 2020

Modified: Eliza Canales - August 2024
"""

from astropy.io import fits
from astropy.wcs import WCS

import numpy as np

import matplotlib.pyplot as plt
import argparse
import bettermoments as bm


def const(X, *c):
    """
    Returns a constant value
    """
    x, y = X
    z = c[0]
    return z


def neg_logL_const(theta, X, data, inv_sigma, log_det_sigma):
    """
    Calculate negative log likelihood for a constant value model,
    including correlated data errors.
    """
    res = data - const(X, *theta)
    neg_logL = 0.5*len(data)*np.log(2.0*np.pi)
    neg_logL += 0.5*log_det_sigma
    neg_logL += 0.5*res.dot(inv_sigma.dot(res))
    return neg_logL


def hess_diag_const(theta, X, inv_sigma):
    """
    Calculate the diagonal of the Hessian matrix for the constant
    value model.
    """
    x, y = X
    d2theta0 = np.sum(inv_sigma)
    return np.array([d2theta0])


def calc_rmse_const(theta, X, data):
    """
    Calculate the root mean square error for the constant
    value model.
    """
    res = data - const(X, *theta)
    return np.sqrt(np.mean(res**2.0))


def plane(X, *c):
    """
    Equation of a plane
    """
    x, y = X
    z = c[0] + c[1]*x + c[2]*y
    return z


def neg_logL_plane(theta, X, data, inv_sigma, log_det_sigma):
    """
    Calculate negative log likelihood for a plane model, including
    correlated data errors.
    """
    res = data - plane(X, *theta)
    neg_logL = 0.5*len(data)*np.log(2.0*np.pi)
    neg_logL += 0.5*log_det_sigma
    neg_logL += 0.5*res.dot(inv_sigma.dot(res))
    return neg_logL


def hess_diag_plane(theta, X, inv_sigma):
    """
    Calculate the diagonal of the Hessian matrix for the plane model.
    """
    x, y = X
    d2theta0 = np.sum(inv_sigma)
    d2theta1 = x.dot(inv_sigma.dot(x))
    d2theta2 = y.dot(inv_sigma.dot(y))
    return np.array([d2theta0, d2theta1, d2theta2])


def calc_rmse_plane(theta, X, data):
    """
    Calculate the root mean square error for the plane model.
    """
    res = data - plane(X, *theta)
    return np.sqrt(np.mean(res**2.0))


def calc_gradient(theta, e_theta, pixsize):
    """
    Calculate gradient and uncertainty
    """
    gradient = 1000.0*np.sqrt(theta[1]**2.0 + theta[2]**2.0)/pixsize
    e_gradient = (1000.0/pixsize)**2.0/gradient * np.sqrt(
        theta[1]**2.0*e_theta[1]**2.0 + theta[2]**2.0*e_theta[2]**2.0)
    return gradient, e_gradient


def calc_pa(theta, e_theta):
    """
    Calculate position angle and uncertainty
    """
    pa = np.rad2deg(np.arctan2(theta[2], -theta[1]) - np.pi/2.0) % 360.0
    e_pa = np.rad2deg(1.0/(theta[1]**2.0 + theta[2]**2.0)) * np.sqrt(
        theta[2]**2.0*e_theta[1]**2.0 + theta[1]**2.0*e_theta[2]**2.0)
    return pa, e_pa


def fit(data,e_data,data3,velax,header):
    """
    Fit a plane to a FITS image.

    Inputs:
      image, e_image :: string
        FITS image with data and uncertainties
      fit_fname :: string
        Where the fit image is saved
    """
    #
    # Get data
    #
    hpbw = header['BMAJ']*3600
    wcs = WCS(header).celestial
    xaxis = np.arange(header['NAXIS1']-1, -1, -1)
    xaxis = xaxis - len(xaxis)/2.0
    yaxis = np.arange(header['NAXIS2'])
    yaxis = yaxis - len(yaxis)/2.0
    ygrid, xgrid = np.meshgrid(yaxis, xaxis, indexing='ij')
    good = (~np.isnan(data)) * (e_data > 0)
    beam_area = (
        np.pi * (hpbw / 3600.0 / header['CDELT2'])**2.0 /
        (4.0 * np.log(2.0)))
    #
    # Prepare for fit
    #
    fit_x = xgrid[good].flatten()
    fit_y = ygrid[good].flatten()
    fit_data = data[good].flatten()
    fit_e_data = e_data[good].flatten()
    nbeam = len(fit_data) / beam_area
    # distance between each pixel and each other pixel
    dist2 = (fit_x[:, None] - fit_x)**2.0 + (fit_y[:, None] - fit_y)**2.0
    # correlation coefficient
    kernel = np.exp(-0.5 * dist2 / (hpbw**2.0/3600.0**2.0/(
        header['CDELT2']**2.0 * 8.0*np.log(2.0))))
    # truncate kernel at FWHM
    kernel[kernel < 0.5] = 0.0
    # covariance matrix
    sigma = fit_e_data[:, None] * fit_e_data * kernel
    inv_sigma = np.linalg.inv(sigma)
    _, log_det_sigma = np.linalg.slogdet(sigma)
    cov_dot_data = np.linalg.solve(sigma, fit_data)
    #
    # Design matrices
    #
    const_design = np.ones((len(fit_x), 1))
    plane_design = np.stack([np.ones(len(fit_x)), fit_x, fit_y], axis=1)
    #
    # Fit constant value model
    #
    cov_dot_design = np.linalg.solve(sigma, const_design)
    popt_const = np.linalg.solve(
        const_design.T.dot(cov_dot_design),
        const_design.T.dot(cov_dot_data))
    perr_const = np.sqrt(
        1.0/hess_diag_const(popt_const, (fit_x, fit_y), inv_sigma))
    bic_const = 1.0*np.log(nbeam) + 2.0*neg_logL_const(
        popt_const, (fit_x, fit_y), fit_data, inv_sigma, log_det_sigma)
    rmse_const = calc_rmse_const(
        popt_const, (fit_x, fit_y), fit_data)
    #
    # Fit plane model
    #
    cov_dot_design = np.linalg.solve(sigma, plane_design)
    popt_plane = np.linalg.solve(
        plane_design.T.dot(cov_dot_design),
        plane_design.T.dot(cov_dot_data))
    perr_plane = np.sqrt(
        1.0/hess_diag_plane(popt_plane, (fit_x, fit_y), inv_sigma))
    bic_plane = 3.0*np.log(nbeam) + 2.0*neg_logL_plane(
        popt_plane, (fit_x, fit_y), fit_data, inv_sigma, log_det_sigma)
    rmse_plane = calc_rmse_plane(
        popt_plane, (fit_x, fit_y), fit_data)
    best_fit = plane((xgrid, ygrid), *popt_plane)
    best_fit[~good] = np.nan
    # calculate R^2
    R2 = 1.0 - np.nansum((data-best_fit)**2.0) / \
        np.nansum((data - np.nanmean(data))**2.0)
    # get gradient and position angle
    gradient, e_gradient = calc_gradient(
        popt_plane, perr_plane, header['CDELT2']*3600.0)
    pa, e_pa = calc_pa(popt_plane, perr_plane)

    return rmse_plane

def main(fnames,moment):
    resolutions = [float(fname.split("_")[2].replace(".fits", "")) for fname in fnames]
    rmses = []
    for fname in fnames:
        rmses.append(fit(fname,moment))
    for i in range(len(resolutions)):
        print("Resolution of",resolutions[i],"arcsecs has an rmse of",rmses[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show R2 value of files",
        prog="fit_gradient.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "fnames",
        type=str,
        nargs="+",
        help="RRL data cubes",
    )
    parser.add_argument(
        "-M",
        "--moment",
        type=int,
        default=0,
        help="Moment used in calculation"
    )
    args = parser.parse_args()
    main(**vars(args))
