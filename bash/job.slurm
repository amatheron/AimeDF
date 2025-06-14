#!/bin/bash
#SBATCH --job-name=33j_factor-015_02in-0_Om-05_Am-05_def-20-2
#SBATCH --partition=hij-gpu
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=600GB
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/33j_factor-015_02in-0_Om-05_Am-05_def-20-2.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/33j_factor-015_02in-0_Om-05_Am-05_def-20-2.err

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield  # activate here the anaconda environment you want to use

python /home/yu79deg/darkfield_p5438/src/darkfield/dfdf_Aime.py -N 8000 --yaml 33j_factor-015_02in-0_Om-05_Am-05_def-20-2.yaml
