#!/bin/bash
#SBATCH -J NAF
#SBATCH -o logs/NAF_%j.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH -p yonitq

INPUT_DIM=${1:-"14"}
FLOW=${2:-"affine"}
DATASET="lhc_binned"
MIN_BIN="2"
MAX_BIN="5"
shift 2

echo "Parameters"
echo "Input dimension: $INPUT_DIM"
echo "Bins: $MIN_BIN to $MAX_BIN"
echo "FLOW: $FLOW"
echo "=================================="

#source /Users/rotem/opt/anaconda3/bin/activate NAF
source /opt/anaconda3/bin/activate NAF
echo "env activated"
for i in 3 4 5 2
do
    EXPERIMENT_NAME="$i"of"$MAX_BIN"
    echo "Running bin $EXPERIMENT_NAME"
    python maf_experiments.py --dataset "$DATASET" --flowtype "$FLOW" --input_dim "$INPUT_DIM" --experiment_name "$EXPERIMENT_NAME"
done