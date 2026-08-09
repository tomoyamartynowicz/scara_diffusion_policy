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
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
RUN_NAME="${RUN_NAME:-h32_a16_bs${BATCH_SIZE}_lr${LEARNING_RATE}_seed${SEED}}"
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

echo "=================================================="
echo "Job:        $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Dataset:    $DATASET_DIR"
echo "Output:     $RUN_DIR"
echo "Batch size: $BATCH_SIZE"
echo "LR:         $LEARNING_RATE"
echo "Seed:       $SEED"
echo "Epochs:     $NUM_EPOCHS"
echo "Started:    $(date)"
echo "=================================================="
nvidia-smi

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv \
  --loop=5 > "$RUN_DIR/gpu_metrics.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

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
  --batch-size "$BATCH_SIZE" \
  --num-workers "$SLURM_CPUS_PER_TASK" \
  --num-epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --seed "$SEED" \
  --save-every 5 \
  "${resume_args[@]}"

echo "=================================================="
echo "Finished: $(date)"
echo "=================================================="
