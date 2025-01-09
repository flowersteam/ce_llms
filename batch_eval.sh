#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --time=0:19:59
#SBATCH --cpus-per-task=24
#SBATCH --array=0-124
#SBATCH -o logs/batch_eval_log_%A_%a.log
#SBATCH -e logs/batch_eval_log_%A_%a.log
#SBATCH -J batch_eval
##SBATCH --qos=qos_gpu_a100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311


seed_paths=(
#  results/*/*/*
  results/scale_small_mixed_dataset_webis_*gen_train_ratio_0.1*/*/*
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
python evaluate_generations.py --emb --gib --tox --pos --input --generations 0 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
#python evaluate_generations.py --gib --input --generations 1 15 16 17 18 19 --seed-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
