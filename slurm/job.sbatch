#!/bin/bash
#SBATCH -J QCNN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00

module load 2025
module load Python/3.13.1-GCCcore-14.2.0

source /gpfs/home6/mvschijndel/QML-Medicine/p311venv/bin/activate

python src/train.py --config ./configs/config.yaml --mode quantum
