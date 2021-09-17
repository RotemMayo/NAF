#!/bin/bash
#SBATCH -J NAF
#SBATCH -o logs/NAF_%j.out
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A rotemov-account
#SBATCH -p allq

INPUT_DIM=${1:-"7"}
EXPERIMENT_NAME=${2:-""}
SP=${3:-"0.1"}
FLOW=${4:-"affine"}
DATASET=${5:-"lhc"}
set --

echo "Parameters"
echo "INPUT_DIM: $INPUT_DIM"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SP: $SP"
echo "FLOW: $FLOW"
echo "=================================="

source /opt/anaconda3/bin/activate NAF
echo "env activated"
python maf_experiments.py --dataset "lhc" --flowtype "$FLOW" --signal "$SP" --input_dim "$INPUT_DIM" --experiment_name "$EXPERIMENT_NAME"
echo "model trained"
python test_model.py
echo "test complete"
echo "Done"
