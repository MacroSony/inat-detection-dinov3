#!/bin/bash
#SBATCH --job-name=triton-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/zihanw/benchmark_%j.out

REPO_DIR="inat-detection-dinov3"
SIF_FILE="$SCRATCH/inat_v1.sif"
SCRATCH_OUT="$SCRATCH/dinov3_output"

mkdir -p $SCRATCH_OUT

apptainer exec --nv --bind $REPO_DIR:/workspace \
                    --bind $SCRATCH:$SCRATCH \
                    --pwd /workspace $SIF_FILE \
                    python train.py fit \
                    --config dinov3_config.yaml \
                    --trainer.max_epochs 1 \
                    --trainer.default_root_dir $SCRATCH_OUT \
                    --model.train_config.output_dir $SCRATCH_OUT