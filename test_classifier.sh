#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --time=01:55:59
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
##SBATCH --array=0-0
#SBATCH -o logs/log_%A_%a.log
#SBATCH -e logs/log_%A_%a.log
#SBATCH --qos=qos_gpu-dev


module load python/3.10.4

conda activate llm_ce

seed=$SLURM_ARRAY_TASK_ID


python3 pol_classifier.py --dataset_name twitter --model Llama-3.1-70B-Instruct-Turbo --n_samples 1000 --seed $seed