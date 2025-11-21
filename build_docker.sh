#!/bin/bash

#SBATCH --ntasks=4
#SBATCH --nodes=4

export MONARCH_FOLDER=${MONARCH_FOLDER:="/home/monarch/"}

srun docker build -f ${MONARCH_FOLDER}/Dockerfile_amd -t monarch_amd ${MONARCH_FOLDER}