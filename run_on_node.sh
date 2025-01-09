#!/bin/bash
#SBATCH -A vgw@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:59
#SBATCH --cpus-per-task=24
##SBATCH --array=0-90
#SBATCH -o logs/run_on_node_log_%A_%a.log
#SBATCH -e logs/run_on_node_log_%A_%a.log
#SBATCH -J run_on_node

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311

python play_with_dataset.py