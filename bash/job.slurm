#!/bin/bash
#SBATCH --job-name=LP_256
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=600GB
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_256.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_256.err

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield  # activate here the anaconda environment you want to use

python /home/yu79deg/darkfield_p5438/src/darkfield/dfdf_Aime.py -N 6000 --yaml LP_256.yaml
