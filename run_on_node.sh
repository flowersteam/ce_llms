#!/bin/bash
#SBATCH -A imi@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
##SBATCH -A imi@cpu
#SBATCH --time=01:59:59
#SBATCH --array=3-3
##SBATCH --array=0-3
#SBATCH -o logs/run_on_node_log_%A_%a.log
#SBATCH -e logs/run_on_node_log_%A_%a.log
##SBATCH --qos=qos_gpu-dev

module load python/3.10.4
module load cuda/12.1.0
module load cudnn/8.9.7.29-cuda
conda activate llm_ce

case $SLURM_ARRAY_TASK_ID in
  0)
    python evaluate_generations.py --experiment-dir dev_results/human_data_ratio_particiapnts_2_generated_dataset_size_1000_human_dataset_size_3000/seed_3_2024-09-05_18-00-24
    ;;
  1)
    python evaluate_generations.py --experiment-dir dev_results/human_data_ratio_particiapnts_2_generated_dataset_size_2000_human_dataset_size_2000/seed_2_2024-09-05_18-00-24
    ;;
  2)
    python evaluate_generations.py --experiment-dir dev_results/human_data_ratio_particiapnts_2_generated_dataset_size_3000_human_dataset_size_1000/seed_1_2024-09-05_18-00-24
    ;;
  3)
    python evaluate_generations.py --experiment-dir dev_results/human_data_ratio_particiapnts_2_generated_dataset_size_4000_human_dataset_size_0/seed_0_2024-09-05_17-58-52
    ;;
  *)
    echo "Error: SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID) is not recognized."
    exit 1
    ;;
esac



