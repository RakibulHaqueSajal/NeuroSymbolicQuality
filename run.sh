#!/bin/bash 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 
#SBATCH --open-mode=append 
#SBATCH --time=0-15:30:00 
#SBATCH --mem=96G 
#SBATCH --partition=general-gpu
#SBATCH --output=log_%J.txt




#/uu/sci.utah.edu/scratch/Rakib_data/miniconda3/usr/sci/scratch/Rakib_data/miniconda3/etc/profile.d/conda.sh
python main.py