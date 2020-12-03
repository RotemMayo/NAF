#!/bin/bash
#SBATCH -J AE
#SBATCH -o logs/AE_%j.out
#SBATCH -N 4
#SBATCH -c 32
#SBATCH --mem=40G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH -p yonitq

source /opt/anaconda3/bin/activate ae
echo "env activated"
python combine_bg_sig_dataset.py "R0.4_all"
echo "dataset combined"
python auto_encoder.py
echo "model trained"
