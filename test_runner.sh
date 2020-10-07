#!/bin/bash
#SBATCH -J NAF
#SBATCH -o logs/NAF_%j.log
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH --partition=yonitq

source /opt/anaconda3/bin/activate NAF
echo "Starting tests"
python test_model.py 
echo "Done" 
