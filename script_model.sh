#!/bin/bash
#SBATCH -c 40
#SBATCH -G 4
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -J NLP
#SBATCH --output=logs_bert.txt

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
python bert.py --sent_encoder sbert

