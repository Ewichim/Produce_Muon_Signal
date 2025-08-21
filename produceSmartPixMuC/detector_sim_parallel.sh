#!/bin/bash
# Run multiple detector simulation jobs in parallel, each with a unique input/output file

N_JOBS=20  # Number of parallel jobs (edit as needed)
benchmark_dir=/home/karri/mucLLPs/mucoll-benchmarks

for i in $(seq 1 $N_JOBS); do
    # Remove previous output if exists
    rm -f output_sim_${i}.slcio
    # Run the simulation in parallel
    ddsim --steeringFile $benchmark_dir/simulation/ilcsoft/steer_baseline.py \
        --inputFile output_gen_${i}.slcio \
        --outputFile output_sim_${i}.slcio &
done

wait

echo "All detector simulation jobs finished." 