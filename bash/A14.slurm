#!/bin/bash
#SBATCH --job-name=darkfield
#SBATCH --partition=hij-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --output=/home/yu79deg/darkfield_p5438/bash/bash_output/A14_darkfield.%j.log
#SBATCH --error=/home/yu79deg/darkfield_p5438/bash/bash_output/A14_darkfield.%j.log

source ~/.bashrc  #  not automatically read by slurm
micromamba activate darkfield

python /home/yu79deg/darkfield_p5438/src/darkfield/$1 -N $2