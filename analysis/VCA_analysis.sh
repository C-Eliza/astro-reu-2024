#!/bin/bash

# Generate simulations with different pixel sizes
#for impix in 50 100 150 200 250 300 350; do
#    python HIIsim.py --impix $impix --fnamebase "test_${impix}"
#done

# Generate velocity statistics
python VCA.py fits/test_*rrl_200.0.fits
