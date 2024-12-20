#!/bin/bash

# Generate seeded simulations with different resolutions
for mach_driving in 0 0.5 1 1.5 2 2.5 3 3.5 4 4.5 5; do
	for seed in 0 1 2 3 4; do
		for resolution in 100.0 200.0 300.0 400.0 500.0 600.0; do
			bettermoments "fits/runs/run_s${seed}_md${mach_driving}rrl_${resolution}.fits" -method zeroth -clip 20
			bettermoments "fits/runs/run_s${seed}_md${mach_driving}rrl_${resolution}.fits" -method first -clip 20
			bettermoments "fits/runs/run_s${seed}_md${mach_driving}rrl_${resolution}.fits" -method second -clip 20
		done
	done
done
