#!/bin/bash

#SBATCH --job-name=scara_dp
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6G
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

PIXI_PROJECT="/scratch/$USER/thesis/envs/diffusion_policy"
PROJECT_DIR="$HOME/scara_ws/src/scara_diffusion_policy"
DATASET_DIR="/scratch/$USER/thesis/datasets/leaf_cutting_experiment_rgb"
RUN_ROOT="/scratch/$USER/thesis/runs/diffusion_policy"

BATCH_SIZE=64
HORIZON=32
N_OBS_STEPS=2
N_ACTION_STEPS=16
NUM_EPOCHS=600
SEED=42
WANDB_RUN_MODE="${WANDB_MODE:-offline}"
SMOKE_TEST="${DP_SMOKE_TEST:-0}"

if [[ "$SMOKE_TEST" != "0" && "$SMOKE_TEST" != "1" ]]; then
    echo "DP_SMOKE_TEST must be 0 or 1, got: $SMOKE_TEST" >&2
    exit 1
fi

RUN_NAME="h${HORIZON}_a${N_ACTION_STEPS}_bs${BATCH_SIZE}_seed${SEED}"
TRAIN_OVERRIDES=()
if [[ "$SMOKE_TEST" == "1" ]]; then
    NUM_EPOCHS=2
    RUN_NAME="smoke_${RUN_NAME}_${SLURM_JOB_ID}"
    TRAIN_OVERRIDES=(
        training.debug=true
        policy.num_inference_steps=2
        task.env_runner.n_test_samples=1
        task.env_runner.batch_size=1
        logging.mode=offline
    )
fi

RUN_DIR="$RUN_ROOT/$RUN_NAME"
LOG_FILE="$RUN_DIR/train_${SLURM_JOB_ID}.log"

if [[ ! -f "$PIXI_PROJECT/pixi.toml" ]]; then
    echo "Pixi manifest not found: $PIXI_PROJECT/pixi.toml" >&2
    exit 1
fi
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "Project directory not found: $PROJECT_DIR" >&2
    exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory not found: $DATASET_DIR" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$TORCH_HOME"
exec > "$LOG_FILE" 2>&1

cd "$PROJECT_DIR"

echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "Dataset:         $DATASET_DIR"
echo "Run directory:   $RUN_DIR"
echo "Batch size:      $BATCH_SIZE"
echo "Horizon:         $HORIZON"
echo "Observation:     $N_OBS_STEPS steps"
echo "Action chunk:    $N_ACTION_STEPS steps"
echo "Epochs:          $NUM_EPOCHS"
echo "Slurm time limit: ${SLURM_TIMELIMIT:-unknown}"
echo "Weights & Biases: $WANDB_RUN_MODE"
echo "Smoke test:       $SMOKE_TEST"
echo "Started:         $(date)"

nvidia-smi

srun pixi run \
    --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -c 'import sys; import cv2, dill, diffusers, h5py, hydra, numpy, omegaconf, torch, torchvision, wandb, zarr; from diffusion_policy.dataset.scara_image_dataset import ScaraImageDataset; from diffusion_policy.workspace.train_diffusion_unet_image_workspace import TrainDiffusionUnetImageWorkspace; assert torch.cuda.is_available(), "CUDA is unavailable"; print("Import preflight passed"); print("python", sys.executable); print("torch", torch.__version__, torch.__file__); print("cuda", torch.version.cuda); print("gpu", torch.cuda.get_device_name(0))'

srun pixi run \
    --manifest-path "$PIXI_PROJECT/pixi.toml" \
    python -u train.py \
    --config-name=train_diffusion_unet_real_image_workspace \
    task=scara_image \
    task.dataset_path="$DATASET_DIR" \
    horizon="$HORIZON" \
    n_obs_steps="$N_OBS_STEPS" \
    dataset_obs_steps="$N_OBS_STEPS" \
    n_action_steps="$N_ACTION_STEPS" \
    dataloader.batch_size="$BATCH_SIZE" \
    dataloader.num_workers="$SLURM_CPUS_PER_TASK" \
    val_dataloader.batch_size="$BATCH_SIZE" \
    val_dataloader.num_workers="$SLURM_CPUS_PER_TASK" \
    training.device=cuda:0 \
    training.seed="$SEED" \
    training.num_epochs="$NUM_EPOCHS" \
    logging.project=scara_diffusion_policy \
    logging.name="$RUN_NAME" \
    logging.mode="$WANDB_RUN_MODE" \
    hydra.run.dir="$RUN_DIR" \
    "${TRAIN_OVERRIDES[@]}"

echo "Finished: $(date)"
