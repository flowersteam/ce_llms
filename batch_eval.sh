#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
##SBATCH -A vgw@a100
##SBATCH -C a100
#SBATCH --gres=gpu:1
##SBATCH -A imi@cpu
#SBATCH --cpus-per-task=24
#SBATCH --time=0:29:59
#SBATCH --array=200-399
##SBATCH --array=0-29
#SBATCH -o logs/batch_eval_log_%A_%a.log
#SBATCH -e logs/batch_eval_log_%A_%a.log
#SBATCH -J batch_eval
##SBATCH --qos=qos_gpu_a100-dev
##SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100

module load python/3.12.2
conda activate eval_312


seed_paths=(
#  webis_clusters_results/*webis*cluster*_1/generated_*_*/*
#  quality_results/*type_Q*_1/generated_*_*/*

#  quality_results/*wikipedia*_1/generated_*_*/*
#  quality_results/*webis*cluster_*_1/generated_*_*/*
#  quality_results/*100m*type_s*_1/generated_*_*/*
#  quality_results/*senator_t*type_s*_1/generated_*_*/*
#  quality_results/*t_submissions*type_s*_1/generated_*_*/*

#  quality_results/*_merged_*_1/generated_*_*/*
  simulation_results/merged_clusters/*_merged_*_participants_1/generated_*_*/*
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

## MERGED ##
#python evaluate_generations.py --emb --cap 160 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
#python evaluate_generations.py --emb --cap 250 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
python evaluate_generations.py --emb --partition webis_reddit --cap 160 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
python evaluate_generations.py --emb --partition 100m_tweets --cap 160 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
python evaluate_generations.py --emb --partition wikipedia --cap 160 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}

#python evaluate_generations.py --emb --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
#python evaluate_generations.py --llama-quality-scale --emb --generations 0 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}

exit
# sequential
counter=0
for path in "${filtered_paths[@]}"; do
    ((counter++))
    echo "Evaluating path #$counter: $path"
    echo "----------------------"
    python evaluate_generations.py --emb --seed-dir "$path"
done