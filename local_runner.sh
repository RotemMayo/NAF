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
EPOCH=${3:-"40"}
BIN=${4:-"2"}
MAX_BIN=${5:-"5"}
DATASET=${6:-"lhc_binned"}
shift 6

echo "Parameters"
echo "Input dimension: $INPUT_DIM"
echo "Bin: $BIN of $MAX_BIN"
echo "FLOW: $FLOW"
echo "=================================="

#source /Users/rotem/opt/anaconda3/bin/activate NAF
source /opt/anaconda3/bin/activate NAF
echo "env activated"
EXPERIMENT_NAME="$BIN"of"$MAX_BIN"
echo "Running training $EXPERIMENT_NAME"
python maf_experiments.py --dataset "$DATASET" --epoch "$EPOCH" --flowtype "$FLOW" --input_dim "$INPUT_DIM" --experiment_name "$EXPERIMENT_NAME"