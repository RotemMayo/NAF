#!/bin/bash
#SBATCH -J NAF
#SBATCH -o logs/NAF_%j.out
#SBATCH -e logs/NAF_%j.err
#SBATCH -N 2
#SBATCH -c 20
#SBATCH --mem=40G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A rotemov-account
#SBTACH -p yonitq,allq

source /opt/anaconda3/bin/activate NAF
cd /usr/people/snirgaz/rotemov/Projects/NAF
python maf_experiments.py --dataset 'lhc' --flowtype 'affine' --signal '0.1'
echo "Done"
