#!/bin/bash
#SBATCH -J NAF_TEST
#SBATCH -o logs/NAF_TEST_%j.log
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH --partition=yonitq

source /opt/anaconda3/bin/activate NAF
echo "Starting tests"
python test_model.py 
echo "Done" 
