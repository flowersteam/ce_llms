#!/bin/bash
##SBATCH -A imi@a100
##SBATCH -C a100
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH -A vgw@h100
#SBATCH -C h100
##SBATCH --qos=qos_gpu_h100-dev
#SBATCH --gres=gpu:3
#SBATCH --time=03:59:59
#SBATCH --array=0-17
#SBATCH -o logs/run_on_node_log_%A_%a.log
#SBATCH -e logs/run_on_node_log_%A_%a.log
#SBATCH -J run_on_node

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311

seed_paths=(dev_results/human_ai_ratio_no_tldr_v2_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_*/*)
#seed_paths=(dev_results/human_ai_ratio_no_tldr_v2_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_8000_Meta-Llama-3.1-8B_participants_1_roof_prob_0.03/generated_*/*)

# Initialize an empty array to store paths that contain gen_19
filtered_paths=()

for path in "${seed_paths[@]}"; do
    # Check if the gen_19 subdirectory exists within the current path
    if [[ -d "$path/gen_19" ]]; then
        # If it exists, add this path to the filtered_paths array
        filtered_paths+=("$path")
    fi
done

echo "Number of paths with gen_19: ${#filtered_paths[@]}"

for part in part_0 part_1 ; do
  # not enough VRAM to compute emb tox and ce at the same time
  python evaluate_generations.py --participant $part --emb --tox --experiment-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
  python evaluate_generations.py --participant $part --ce --emb --tox --experiment-dir ${filtered_paths[$SLURM_ARRAY_TASK_ID]}
done