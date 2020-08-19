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
#SBTACH -p yonitq
source /opt/anaconda3/bin/activate NAF
# eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
# conda activate NAF
echo "maf_experiments.py --dataset $dataset"
python maf_experiments.py --dataset $dataset
echo "Done"
