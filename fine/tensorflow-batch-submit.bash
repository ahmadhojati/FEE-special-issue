#!/bin/bash

#Account and Email Information
#SBATCH -A ahojatimalekshah
#SBATCH --mail-type=end
#SBATCH --mail-user=ahmadhojatimalek@boisestate.edu

#SBATCH -J PYTHON          # job name
#SBATCH -o output/results.o%j # output and error file name (%j expands to jobID)
#SBATCH -e output/errors.e%j
#SBATCH -n 1               # Run one process
#SBATCH --cpus-per-task=28 # allow job to multithread across all cores
#SBATCH -p gpu            # queue (partition) -- defq, ipowerq, eduq, gpuq.
#SBATCH -t 1-00:00:00      # run time (d-hh:mm:ss)
ulimit -v unlimited
ulimit -s unlimited
ulimit -u 1000

module load cuda10.0/toolkit/10.0.130 # loading cuda libraries/drivers 
module load python36          # loading python environment
module load gdal/gcc8/3.0.4	# loading gdal 

python3 Deep_Learning_V2.py
