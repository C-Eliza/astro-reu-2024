"""
gen_moments.py
Creates HII regions, saves their moments and power spectrum
Eliza Canales
"""

import numpy as np
import astropy.units as u
import astropy.constants as c
import argparse
import astropy.io.fits as fits
from HIIsim import Simulation, HIIRegion, split_observations_no_save
from gen_turbulence import gen_turbulence
from turbustat.statistics import PowerSpectrum
import bettermoments as bm

def main(
        impix,
        nchan,
        mean_density,
        mach_number,
        driving_parameter,
        beam_fwhm,
        noise,
        fnamebase,
        constant_density,
        use_GPU,
        seed
        ):

    for s in seed:
        print("Seed: "+str(s))
        for m in mach_number:
            print("Mach: "+str(m))
            # Synthetic observation
            dens1, vel1 = gen_turbulence(
                impix,
                mean_density=mean_density * u.cm**-3,
                seed=s,
                mach_number=m,
                driving_parameter=driving_parameter,
            )

            if constant_density:
                dens1 = mean_density * np.ones(np.shape(dens1)) * u.cm**-3

            region1 = HIIRegion(
                electron_density=dens1,
                velocity=vel1,
                distance=0.25 * u.kpc,
            )
            pixel_size = 50 * u.arcsec / (impix / 50) / (region1.distance / 0.25 / u.kpc)
            obs1 = Simulation(
                nchan=nchan,
                npix=dens1.shape[0],
                lospix=dens1.shape[2],
                pixel_size=pixel_size,
                channel_size=40 * u.kHz,
            )
                                                                                 
            sim_hdul = obs1.simulate(region1, use_GPU=use_GPU, rrl_only = True, no_save=True,filename="",quiet=True)
            for bf in beam_fwhm:
                filename = "fits/"+fnamebase+"_s"+str(s)+"_md"+str(m)+"rrl_"+str(bf)
                analyze(sim_hdul, obs1, bf*u.arcsec, noise*u.K, filename)

def save_moment(moment, dmoment, prev_header, filename, method):
    '''
    Saves a moment map and its uncertainty from bettermoments to storage
    without having saved the source radio image first.

    Inputs:
    moment -- Moment map to be saved.
    dmoment -- Moment uncertainty map to be saved.
    prev_header -- Header of the original simulated data.
    filename -- Basis of the filename for the saved data.
    method -- Method used to make moment map, used for filename and units.

    Outputs:
    None
    '''
    hdul = fits.HDUList()
    if(method=="zeroth"):
        hdul.append(fits.PrimaryHDU(moment.to(u.K*u.km/u.s).value,prev_header))
        hdul[0].header["CRVAL3"] = None
        del hdul[0].header["CDELT3"]
        del hdul[0].header["CTYPE3"]
        del hdul[0].header["CRPIX3"]
        del hdul[0].header["CUNIT3"]
        hdul[0].header["BTYPE"] = "Integrated Brightness Temp"
        hdul.writeto(filename+"_M0.fits",overwrite=True)
        hdul[0].data = dmoment.to(u.K*u.km/u.s).value
        hdul.writeto(filename+"_dM0.fits",overwrite=True)

    if(method=="first"):
        hdul.append(fits.PrimaryHDU(moment.to("km/s").value,prev_header))
        del hdul[0].header["CRVAL3"]
        del hdul[0].header["CDELT3"]
        del hdul[0].header["CTYPE3"]
        del hdul[0].header["CRPIX3"]
        del hdul[0].header["CUNIT3"]
        hdul[0].header["BTYPE"] = "Average Velocity"
        hdul[0].header["BUNIT"] = "km/s"
        hdul.writeto(filename+"_M1.fits",overwrite=True)
        hdul[0].data = dmoment.to("km/s").value
        hdul.writeto(filename+"_dM1.fits",overwrite=True)

    if(method=="second"):
        hdul.append(fits.PrimaryHDU(moment.to("km/s/K").value,prev_header))
        del hdul[0].header["CRVAL3"]
        del hdul[0].header["CDELT3"]
        del hdul[0].header["CTYPE3"]
        del hdul[0].header["CRPIX3"]
        del hdul[0].header["CUNIT3"]
        hdul[0].header["BTYPE"] = "Distribution Width"
        hdul[0].header["BUNIT"] = "km/s"
        hdul.writeto(filename+"_M2.fits",overwrite=True)
        hdul[0].data = dmoment.to("km/s/K").value
        hdul.writeto(filename+"_dM2.fits",overwrite=True)

def analyze(sim_hdul, simul, beam_fwhm, noise, filename):

    '''
    Takes in simulated and convoluted data, creates moment maps for them,
    saves the moment maps, does the power spectrum analysis, and saves it.

    Inputs:
    sim_hdul -- HDUList that contains simulated data, without any noise
    applied or beam convolution.
    simul -- Contains the simulation parameters and inital 3d conditions.
    beam_fwhm -- The full-width half max for the desired radio beam.
    noise -- In brightness temperature, the wanted noise level after
    convolution.
    filename -- Basis of the filename for the saved data.

    Outputs: None
    '''

    obs_hdul = split_observations_no_save(sim_hdul, beam_fwhm, noise)

    hpbw = obs_hdul[0].header['BMAJ'] * 3600
    rms = bm.estimate_RMS(data=obs_hdul[0].data, N=5)
    velax = simul.velo_axis

    moment0, dmoment0 = bm.collapse_zeroth(velax=velax,rms=rms,data=obs_hdul[0].data)
    moment0 *= u.K
    dmoment0 *= u.K
    moment1, dmoment1 = bm.collapse_first(velax=velax,rms=rms,data=obs_hdul[0].data)
    moment2, dmoment2 = bm.collapse_second(velax=velax,rms=rms,data=obs_hdul[0].data)
    moment2 /= u.K
    dmoment2 /= u.K

    moment0[moment0<3*rms*simul.nchan*u.K*u.km/u.s] = np.nan
    moment1[np.isnan(moment0)] = np.nan
    moment2[np.isnan(moment0)] = np.nan
    dmoment0[np.isnan(moment0)] = np.nan
    dmoment1[np.isnan(moment0)] = np.nan
    dmoment2[np.isnan(moment0)] = np.nan

    save_moment(moment0,dmoment0,obs_hdul[0].header,filename,"zeroth")
    save_moment(moment1,dmoment1,obs_hdul[0].header,filename,"first")
    save_moment(moment2,dmoment2,obs_hdul[0].header,filename,"second")

    #Power spectrum
    pspec0 = PowerSpectrum(moment0.to("K km/s").value, header=obs_hdul[0].header)
    pspec1 = PowerSpectrum(moment1.to("km/s").value, header=obs_hdul[0].header)
    pspec2 = PowerSpectrum(moment2.to("km/s/K").value, header=obs_hdul[0].header)
    pspec0.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec1.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec2.run(xunit=u.arcsec**-1,high_cut=1/(hpbw*u.arcsec))
    pspec0.save_results(filename+'_M0')
    pspec1.save_results(filename+'_M1')
    pspec2.save_results(filename+'_M2')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculates the powerlaw spectrum and saves it",
        prog="getproperties.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--impix",
        type=int,
        default=256,
        help="Simulation pixel dimensions (cubical)",
    )
    parser.add_argument(
        "--nchan",
        type=int,
        default=256,
        help="Number of frequency channels",
    )
    parser.add_argument(
        "--mean_density",
        type=float,
        default=1000.0,
        help="Mean electron density (cm-3)",
    )
    parser.add_argument(
        "--mach_number",
        type=float,
        nargs="+",
        default=[5.0],
        help="Mach number",
    )
    parser.add_argument(
        "--driving_parameter",
        type=float,
        default=1,
        help="Driving parameter",
    )
    parser.add_argument(
        "--beam_fwhm",
        type=float,
        nargs="+",
        default=[200.0],
        help="Beam FWHM (arcsec)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Noise (K)",
    )
    parser.add_argument(
        "--fnamebase",
        type=str,
        default="region1",
        help="Filename base name",
    )
    parser.add_argument(
        "--constant_density",
        action="store_true",
        help="Consant region density",
    )
    parser.add_argument(
        "--use_GPU",
        action="store_true",
        help="Use GPU for simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[100],
        help="Seed for the simulation"
    )
    args = parser.parse_args()
    main(**vars(args))
