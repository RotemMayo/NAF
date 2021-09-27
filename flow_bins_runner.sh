#!/bin/bash
FLOW=${1:-"affine"}
shift 1

for i in 5 6 7 4 3
do
    /usr/bin/sbatch local_runner.sh 4 "$FLOW" 40 "$i" 7 "lhc_binned"
done