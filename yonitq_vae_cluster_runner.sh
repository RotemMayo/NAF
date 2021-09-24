#!/bin/bash
#SBATCH -J VAE
#SBATCH -o logs/VAE_%j.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=10G
#SBATCH --mail-type=END
#SBATCH --mail-user=rotem.ovadia@mail.huji.ac.il
#SBATCH -A yonit-account
#SBATCH -p yonitq

NSIG=${1:-"100000"}
shift 1

echo "Parameters"
echo "NSIG: $NSIG"
echo "=================================="

source /opt/anaconda3/bin/activate NAF
echo "env activated"
python beta1_vae_model.py "$NSIG"
echo "model trained"
echo "Done"
