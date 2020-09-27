#!/bin/bash
#SBATCH -J NAF
#SBATCH -o logs/NAF_%j.out
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A rotemov-account
#SBTACH -p yonitq,allq

INPUT_DIM=${1:-"7"}
EXPERIMENT_NAME=${2:-""}
set --

source /opt/anaconda3/bin/activate NAF
cd /usr/people/snirgaz/rotemov/Projects/NAF
python maf_experiments.py --dataset 'lhc' --flowtype 'affine' --signal '0.1' --input_dim "$INPUT_DIM" --experiment_name "$EXPERIMENT_NAME"
echo "Done"
