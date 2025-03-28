#!/bin/bash
##SBATCH -A imi@h100
##SBATCH -C h100
##SBATCH -A vgw@a100
##SBATCH -C a100
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:1
#SBATCH -A imi@cpu
#SBATCH --time=01:59:59
#SBATCH --array=25-119
#SBATCH --cpus-per-task=24
#SBATCH -o logs/run_webis_clusters_%A_%a.log
#SBATCH -e logs/run_webis_clusters_%A_%a.log
#SBATCH -J run_webis_clusters
##SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate clustering_311

clustering_paths=(
  viz_results/webis_miniclusters_merge_80k/results/*
)

# Initialize an empty array to store paths that contain gen_19
echo "Number of paths: ${#clustering_paths[@]}"

# Parameters to iterate over
path=${clustering_paths[$SLURM_ARRAY_TASK_ID]}

python evaluate_webis_clusters.py $path