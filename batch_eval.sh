#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
##SBATCH -A vgw@a100
##SBATCH -C a100
#SBATCH --gres=gpu:1
##SBATCH -A imi@cpu
#SBATCH --cpus-per-task=24
#SBATCH --time=0:09:59
#SBATCH --array=0-199
#SBATCH -o logs/batch_eval_log_%A_%a.log
#SBATCH -e logs/batch_eval_log_%A_%a.log
#SBATCH -J batch_eval
##SBATCH --qos=qos_gpu_a100-dev
##SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311


seed_paths=(
  webis_clusters_results/*webis*cluster*_1/generated_*_*/*
#  quality_results/*type_Q*_1/generated_*_*/*
#  quality_results/*webis*cluster_*_1/generated_*_*/*
#  quality_results/*100m*type_s*_1/generated_*_*/*
#  quality_results/*senator_t*type_s*_1/generated_*_*/*
#  quality_results/*t_submissions*type_s*_1/generated_*_*/*
)


filtered_paths=()

for path in "${seed_paths[@]}"; do
    if [[ -f "$path/gen_19/part_0/log.json" ]]; then
        # If it exists, add this path to the filtered_paths array
        filtered_paths+=("$path")
    fi
done

#seed_paths=(dev_results_scale/*tweet*pants_*/generated_*/*)

# Initialize an empty array to store paths that contain gen_19
echo "Number of paths: ${#filtered_paths[@]}"
#python evaluate_generations.py --gib --tox --pos --input --generations 0 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
#python evaluate_generations.py --emb --generations 0 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
python evaluate_generations.py --llama-quality-scale --emb --generations 0 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
