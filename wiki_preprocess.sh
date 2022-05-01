#!/bin/bash
#SBATCH -c 40
#SBATCH -G 1
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END
#SBATCH -J NLP
#SBATCH --output=logs_bert.txt

cd
source .bashrc
cd ~/cross_domain_coherence
conda activate pytorch
echo "processing wiki_bigram_Artist"
python preprocess.py --data_name  wiki_bigram_Artist
echo "processing wiki_bigram_Athlete"
python preprocess.py --data_name  wiki_bigram_Athlete
echo "processing wiki_bigram_Politician"
python preprocess.py --data_name  wiki_bigram_Politician
echo "processing wiki_bigram_Writer"
python preprocess.py --data_name  wiki_bigram_Writer
echo "processing wiki_bigram_MilitaryPerson"
python preprocess.py --data_name  wiki_bigram_MilitaryPerson
echo "processing wiki_bigram_OfficeHolder"
python preprocess.py --data_name  wiki_bigram_OfficeHolder
echo "processing wiki_bigram_Scientist"
python preprocess.py --data_name  wiki_bigram_Scientist
echo "processing wiki_bigram_Plant"
python preprocess.py --data_name  wiki_bigram_Plant
echo "processing wiki_bigram_CelestialBody"
python preprocess.py --data_name  wiki_bigram_CelestialBody
echo "processing wiki_bigram_EducationalInstitution"
python preprocess.py --data_name  wiki_bigram_EducationalInstitution
echo "processing wiki_bigram_easy"
python preprocess.py --data_name  wiki_bigram_easy
