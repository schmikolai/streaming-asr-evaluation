#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=08:00:00
#SBATCH --exclude=marvin,deepthought,glados,jarvis
#SBATCH --output=/path/to/logs/%x_%j.out
#SBATCH --error=/path/to/logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ns115@hdm-stuttgart.de

# Check if conda is available *and* initialized
if ! type conda &> /dev/null || ! conda info &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi

conda activate asr

cd "$(dirname "$0")"

python -u run.py