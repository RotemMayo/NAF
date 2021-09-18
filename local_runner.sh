#!/bin/bash
INPUT_DIM=${1:-"14"}
FLOW=${2:-"affine"}
DATASET="lhc_binned"
MIN_BIN="5"
MAX_BIN="10"
shift 2

echo "Parameters"
echo "Input dimension: $INPUT_DIM"
echo "Bins: $MIN_BIN to $MAX_BIN"
echo "FLOW: $FLOW"
echo "=================================="

source /Users/rotem/opt/anaconda3/bin/activate NAF
echo "env activated"
for i in 5 6 7 8 9 10
do
    EXPERIMENT_NAME="$i"of"$MAX_BIN"
    echo "Running bin $EXPERIMENT_NAME"
    python maf_experiments.py --dataset "$DATASET" --flowtype "$FLOW" --input_dim "$INPUT_DIM" --experiment_name "$EXPERIMENT_NAME"
done