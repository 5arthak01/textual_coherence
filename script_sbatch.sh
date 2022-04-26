#!/bin/bash
#SBATCH --cpus-per-task=40
#SBATCH --gpus=2
#SBATCH --time=3-00:00:00
#SBATCH --output=experiment_logs.txt
#SBATCH --mail-type=END

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
python run_bigram_coherence.py --data_name wsj_bigram --sent_encoder average_glove
