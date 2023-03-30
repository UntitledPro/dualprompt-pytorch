#!/bin/bash

#SBATCH --job-name=dualprompt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -p batch
#SBATCH -w agi1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=20G
#SBATCH --time=14-0
#SBATCH -o %N_%x_%j.out
#SBTACH -e %N_%x_%j.err

python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env main.py \
        cifar100_dualprompt \
        --model vit_base_patch16_224 \
        --batch-size 24 \
        --data-path ~/workspace/datasets/ \
        --output_dir ./output
