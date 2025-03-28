#!/bin/bash
#SBATCH -A vgw@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --time=05:59:59
#SBATCH --cpus-per-task=24
#SBATCH --array=0-1
#SBATCH -o logs/visualize_datasets_%A_%a.log
#SBATCH -e logs/visualize_datasets_%A_%a.log
#SBATCH -J visualize_datasets


source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311

# Define the pattern for directories
dirs_pattern=""
dirs_pattern+=" results/scale_unbalanced_sampling_small_mixed_dataset_webis_*_1"
#dirs_pattern+=" old_results/human_ai_ratio_dataset_webis_reddit_split_test_acc_1_ft_size_4000_mixed_participants_1"

#xdg-open eval_results/results/scale_unbalanced_sampling_small_mixed_dataset_webis_reddit_type_standard_presampled_split_all_acc_1_ft_size_4000_mixed_participants_1/visualizations/stella/pca/seed_1_Gen\:2000_of_4000.svg
#xdg-open eval_results/old_results/human_ai_ratio_dataset_webis_reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/visualizations/stella/pca/seed_1_Gen:500_of_4000.svg

matching_dirs=($dirs_pattern)  # Collect matching directories into an array
echo "Found ${#matching_dirs[@]} directories matching the pattern"

# Iterate over all matching directories
dir=${matching_dirs[$SLURM_ARRAY_TASK_ID]}
echo "Processing directory: $dir"
python visualize_datasets.py "$dir"
