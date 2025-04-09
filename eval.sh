#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --exclude=marvin,deepthought,glados,jarvis
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ns115@hdm-stuttgart.de

conda activate asr

python -u run.py