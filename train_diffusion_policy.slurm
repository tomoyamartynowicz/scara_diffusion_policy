#!/bin/bash

#SBATCH --job-name=scara_diffusion
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --account=education-ME-msc-ro
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module purge
module load slurm

export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/scara_ws/src/diffusion_policy:$HOME/scara_ws/src"

PIXI_PROJECT="/scratch/$USER/thesis/envs/diffusion_policy"
DATASET_DIR="/scratch/$USER/thesis/datasets/leaf_cutting_experiment_rgb"
RUN_DIR="/scratch/$USER/thesis/runs/diffusion_policy/h32_a16_bs16_seed42"

export TORCH_HOME="/scratch/$USER/thesis/cache/torch"
export HF_HOME="/scratch/$USER/thesis/cache/huggingface"
export PIXI_CACHE_DIR="/scratch/$USER/thesis/cache/pixi"
export TMPDIR="/scratch/$USER/thesis/tmp/$SLURM_JOB_ID"

mkdir -p "$RUN_DIR" "$TORCH_HOME" "$HF_HOME" "$PIXI_CACHE_DIR" "$TMPDIR"
exec >> "$RUN_DIR/train.log" 2>&1

cd "$HOME/scara_ws/src"

test -f "$PIXI_PROJECT/pixi.toml"
test -d "$DATASET_DIR"
test -f "$HOME/scara_ws/src/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py"

echo "=================================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Dataset:    $DATASET_DIR"
echo "Output:     $RUN_DIR"
echo "Started:    $(date)"
echo "=================================================="
nvidia-smi

resume_args=()
if [[ -f "$RUN_DIR/training_state_last.pt" ]]; then
  echo "Resuming the last completed epoch"
  resume_args+=(--resume)
elif [[ -f "$RUN_DIR/train_config.json" ]]; then
  echo "Incomplete run: train_config.json exists without training_state_last.pt"
  echo "Move this run directory aside or choose a new RUN_DIR before retrying."
  exit 2
fi

srun --mpi=pmix pixi run \
  --manifest-path "$PIXI_PROJECT/pixi.toml" \
  python -u -m scara_diffusion_policy.train \
  --dataset-dir "$DATASET_DIR" \
  --ckpt-dir "$RUN_DIR" \
  --horizon 32 \
  --n-obs-steps 2 \
  --n-action-steps 16 \
  --num-inference-steps 8 \
  --batch-size 16 \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --num-epochs 50 \
  --save-every 5 \
  "${resume_args[@]}"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
