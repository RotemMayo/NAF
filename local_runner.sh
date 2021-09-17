#!/bin/bash
for i in 5 6 7 8 9 10
do
    sh cluster_runner.sh 14 "$i"of10 0.1 affine lhc_binned
done