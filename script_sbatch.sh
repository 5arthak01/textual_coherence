#!/bin/bash
#SBATCH -c 40
#SBATCH -G 1
#SBATCH --time=5-00:00:00
#SBATCH --output=logs_experiments.txt
#SBATCH --mail-type=END

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
python experiments.py
