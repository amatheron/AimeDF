#!/bin/bash
#SBATCH --job-name=LP_83
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=600GB
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_83.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_83.err

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield  # activate here the anaconda environment you want to use

python /home/yu79deg/darkfield_p5438/src/darkfield/dfdf_Aime.py -N 2000 --yaml LP_83.yaml
