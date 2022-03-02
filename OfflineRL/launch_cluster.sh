
#! /bin/bash

for seed in 0 1 2 3 4; do
  sbatch --export=ALG=0,SEED=$seed,DEV=1 run_sbatch.sbatch
  sleep 1
  sbatch --export=ALG=1,SEED=$seed,DEV=1 run_sbatch.sbatch
  sleep 1
done
