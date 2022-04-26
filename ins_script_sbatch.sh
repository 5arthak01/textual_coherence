#!/bin/bash
#SBATCH -c 40
#SBATCH -G 1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -J experiments_ins
#SBATCH --output=logs_experi_ins.txt

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
python experiments_ins.py
