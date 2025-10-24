#!/bin/bash
#SBATCH --job-name=LP_610
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=600GB
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_610.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/LP_610.err
#SBATCH --chdir=/home/yu79deg/darkfield_p5438   # ensure a known working dir

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield  # activate here the anaconda environment you want to use

python /home/yu79deg/darkfield_p5438/src/darkfield/VIBE.py -N 1000 --yaml /home/yu79deg/darkfield_p5438/yamls/LP_610.yaml
