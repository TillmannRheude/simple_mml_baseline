#!/bin/bash

#SBATCH --job-name=SimBas
#SBATCH -p compute
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --output=kinetics_extractor.out

python3 kinetics.py