#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --exclude=marvin,deepthought,glados,jarvis
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ns115@hdm-stuttgart.de

source ~/miniconda3/etc/profile.d/conda.sh
conda activate asr

cd "$(dirname "$0")"

python -u run.py "$@"