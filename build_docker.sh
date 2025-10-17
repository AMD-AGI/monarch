#!/bin/bash

#SBATCH --ntasks=6

#SBATCH --nodes=6

srun docker build -f /mnt/models/mreso/monarch/Dockerfile_amd -t monarch_amd /mnt/models/mreso/monarch/
# srun docker pull docker.io/rocm/megatron-lm:v25.6_py312