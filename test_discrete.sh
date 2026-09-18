#!/bin/bash
#SBATCH --job-name=discrete
#SBATCH --partition=amdfast
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --output=discrete.out

set -euo pipefail

ml PyTorch/2.10.0-foss-2025b-CUDA-12.9.1
source .venv/bin/activate

python3 -u test_all_networks.py --n-seeds 5 --networks mlp,deep_mlp --n-epochs 50
