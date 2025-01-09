#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --time=04:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o logs/vllm_eval_log_%A.log
#SBATCH -e logs/vllm_eval_log_%A.log
#SBATCH -J vllm_eval

start_time=$(date +%s)

# find paths
#seed_paths=(results/human_ai_ratio_dataset_webis_reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/*/* results/human_ai_ratio_dataset_senator_tweets_split_all_acc_1_ft_size_4000_mixed_participants_1/*/*)
#seed_paths=(
#)
seed_paths=(
#  results/scale_small_mixed*/*/*
#  results/scale_unbalanced_sampling_small_mixed*/*/*
#  results/human_ai_ratio_dataset_*/*/*
#  results/scale_small_mixed*gen_train_ratio_0.1*20/*/*
  results/scale_unbalanced_sampling_small_mixed*_1/generated_*_*/*
  results/human_ai_ratio_dataset_webis_reddit_type_*q*_1/generated_*_*/*
  results/human_ai_ratio_dataset_*_1/*/*
)

#seed_paths=(results/human_ai_ratio_dataset_100m_tweets*/generated_*/*)

# Initialize an empty array to store paths that contain gen_19
filtered_paths=()
for path in "${seed_paths[@]}"; do
    if [[ -f "$path/gen_19/part_0/log.json" ]]; then
        # If it exists, add this path to the filtered_paths array
        filtered_paths+=("$path")
    fi
done


echo "Number of paths: ${#filtered_paths[@]}"

# launch server
source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.10.4
conda activate vllm

echo "Launching server"

SERVERLOG="logs/vllm_server_log_$SLURM_JOB_ID"
echo "Server log : $SERVERLOG"

# for llama first launch vllm server with
export VLLM_CONFIGURE_LOGGING=0 # no logging
export VLLM_WORKER_MULTIPROC_METHOD=spawn
vllm serve /lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/05917295788658563fd7ef778b6240ad9867d6d1/ \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.98 \
    --served-model-name llama \
    --dtype half \
    --disable-log-requests \
    --disable-log-stats \
    --enable-prefix-caching \
    --tensor-parallel-size 4 &> "$SERVERLOG" &

# wait for server to load
echo "Loading server"
while true; do
  # Attempt to connect
  response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/models)

  if [ "$response" -eq 000 ]; then
    echo "vllm server not loaded. Retrying in 15 seconds..."
    sleep 15
  else
    echo "vllm server running successfully!"
    break
  fi
done

# launch clients
echo "Launching clients"
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311

max_parallel=30  # Maximum number of parallel processes
current_parallel=0  # Counter for active processes

# subshell
(

for i in "${!filtered_paths[@]}"; do
  python evaluate_generations.py --llama-quality --input --seed-dir ${filtered_paths[$i]}  --generations 0 1 15 16 17 18 19 &
#  python evaluate_generations.py --llama-quality --input --seed-dir ${filtered_paths[$i]}  &
  ((current_parallel++))
  echo "Current parallel" $current_parallel

  if (( current_parallel >= max_parallel )); then
    echo "Parallel batch launched"
    wait
    echo "Parallel batch done"
    current_parallel=0
  fi
done

wait
echo "Clients done"
)

end_time=$(date +%s)
# Calculate the difference
duration=$(( end_time - start_time ))

# Calculate hours, minutes, and seconds
hours=$(( duration / 3600 ))
minutes=$(( (duration % 3600) / 60 ))
seconds=$(( duration % 60 ))

# Format the output to hh:mm:ss
printf -v elapsed_time "%02d:%02d:%02d" $hours $minutes $seconds

echo "Elapsed time: $elapsed_time " 2>&1 | tee -a $log_path
