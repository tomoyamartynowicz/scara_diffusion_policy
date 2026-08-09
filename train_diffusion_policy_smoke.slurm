#!/bin/bash

#SBATCH --job-name=scara_dp_smoke
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
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
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_NAME="${RUN_NAME:-benchmark_h32_a16_bs${BATCH_SIZE}}"
RUN_DIR="/scratch/$USER/thesis/runs/diffusion_policy/$RUN_NAME"

export TORCH_HOME="/scratch/$USER/thesis/cache/torch"
export HF_HOME="/scratch/$USER/thesis/cache/huggingface"
export PIXI_CACHE_DIR="/scratch/$USER/thesis/cache/pixi"

mkdir -p "$RUN_DIR" "$TORCH_HOME" "$HF_HOME" "$PIXI_CACHE_DIR"
exec >> "$RUN_DIR/train.log" 2>&1

cd "$HOME/scara_ws/src"

test -f "$PIXI_PROJECT/pixi.toml"
test -d "$DATASET_DIR"
test -f "$HOME/scara_ws/src/diffusion_policy/diffusion_policy/model/diffusion/conditional_unet1d.py"

echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Dataset:    $DATASET_DIR"
echo "Output:     $RUN_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Started:    $(date)"
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv \
  --loop=5 > "$RUN_DIR/gpu_metrics.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

srun --mpi=pmix pixi run \
  --manifest-path "$PIXI_PROJECT/pixi.toml" \
  python -u -m scara_diffusion_policy.train \
  --dataset-dir "$DATASET_DIR" \
  --ckpt-dir "$RUN_DIR" \
  --num-epochs 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --image-width 320 \
  --image-height 240 \
  --save-every 1 \
  --max-train-batches 20 \
  --max-val-batches 5

echo "Finished: $(date)"
