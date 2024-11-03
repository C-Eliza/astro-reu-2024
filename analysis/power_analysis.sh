#!/bin/bash

# Generate simulations with different pixel sizes
#for impix in 50 100 150 200 250 300 350; do
#    python HIIsim.py --impix $impix --fnamebase "test_${impix}"
#done

# Generate velocity statistics
python powerlaw.py -M 0 fits/test_*rrl_200.0.fits
python powerlaw.py -M 1 fits/test_*rrl_200.0.fits
python powerlaw.py -M 2 fits/test_*rrl_200.0.fits
