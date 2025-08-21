#!/bin/bash
# Run multiple particle gun jobs in parallel, each with a unique output file and random seed

N_JOBS=20  # Number of parallel jobs (edit as needed)
EVENTS=90000 # Number of events per job (edit as needed)
benchmark_dir=/home/karri/mucLLPs/mucoll-benchmarks

for i in $(seq 1 $N_JOBS); do
    # Set a unique random seed for each job
    SEED=$((10500 + i))
    OUTFILE=output_gen_${i}.slcio
    rm -f $OUTFILE
    python $benchmark_dir/generation/pgun/pgun_lcio.py \
        -s $SEED \
        -e $EVENTS \
        --pdg 13 -13 \
        --p 1 100 \
        --theta 10 170 \
        --dz 0 0 1.5 \
        --d0 0 0 0.0009 \
        -- $OUTFILE &
done

wait

echo "All particle gun jobs finished." 