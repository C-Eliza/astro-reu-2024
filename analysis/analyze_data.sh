#!/bin/bash

# Generate seeded simulations with different resolutions
for mach_driving in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
	for seed in 0 1 2 3 4; do
		python HIIsim.py --rrl_only --use_GPU --impix 200 --beam_fwhm 100 200 300 400 500 600 --driving_parameter 1 --mach_number ${mach_driving} --seed ${seed} --fnamebase "runs/run_s${seed}_md${mach_driving}"
	done
done
