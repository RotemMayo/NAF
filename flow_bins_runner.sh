#!/bin/bash
FLOW=${1:-"affine"}
shift 1

for i in 4 5 6 7 8
do
    /usr/bin/sbatch local_runner.sh 14 "$FLOW" 40 "$i" 8 "lhc_binned"
done