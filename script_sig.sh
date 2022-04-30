#!/bin/bash
#SBATCH -c 40
#SBATCH -G 1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -J experiments_dis
#SBATCH --output=logs_sigmoid.txt

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
python sigmoid.py

