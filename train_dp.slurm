#!/bin/bash

#SBATCH --job-name=scara_dp
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

set -euo pipefail

module purge
module load slurm

export PATH="$HOME/.pixi/bin:$PATH"
export TORCH_HOME="/scratch/$USER/thesis/cache/torch"
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export SCARA_DATASET_DIR="/scratch/$USER/thesis/datasets/leaf_cutting_experiment_rgb"
export SCARA_RUN_ROOT="/scratch/$USER/thesis/runs/diffusion_policy"

PIXI_PROJECT="/scratch/$USER/thesis/envs/diffusion_policy"
PROJECT_DIR="$HOME/scara_ws/src/scara_diffusion_policy"

if [[ ! -f "$PIXI_PROJECT/pixi.toml" ]]; then
    echo "Pixi manifest not found: $PIXI_PROJECT/pixi.toml" >&2
    exit 1
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -d "$SCARA_DATASET_DIR" ]]; then
    echo "Dataset directory not found: $SCARA_DATASET_DIR" >&2
    exit 1
fi

mkdir -p "$SCARA_RUN_ROOT" "$TORCH_HOME"

cd "$PROJECT_DIR"

srun pixi run \
    --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -u train.py \
    --config-name=train_diffusion_unet_scara_workspace
