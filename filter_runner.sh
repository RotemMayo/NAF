#!/bin/bash
#SBATCH -J NAF_FILTER
#SBATCH -o logs/NAF_FILTER_%j.log
#SBATCH -N 2
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH --partition=yonitq

source /opt/anaconda3/bin/activate NAF
echo "Filtering by loss"
python loss_filter.py 
echo "Done" 
