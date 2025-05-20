#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH -A imi@h100
##SBATCH --array=0-4 # D
##SBATCH --array=5-9 # D
##SBATCH --array=10-14 # D
#SBATCH --array=15-19 # R
##SBATCH --array=5-19
##SBATCH --array=0-19
#SBATCH --time=1:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o logs/add_qualities_parallel_log_%A_%a.client.log
#SBATCH -e logs/add_qualities_parallel_log_%A_%a.client.log
#SBATCH -J add_qualities_parallel
#SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.12.2
conda activate vllm_312

# launch server
echo "Launching server"
# for llama first launch vllm server with
#export VLLM_CONFIGURE_LOGGING=0 # no logging
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve /lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/ \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.95 \
    --served-model-name llama \
    --dtype auto \
    --enable-prefix-caching  &> logs/add_qualities_parallel_log_$SLURM_JOB_ID_$SLURM_ARRAY_TASK_ID.server.log &

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
module load python/3.12.2
conda activate cluster_312

#python prepare_100m_tweets_dataset.py
#python prepare_reddit_submissions_dataset.py
#python eval_openmeva.py

#python add_qualities_to_dataset.py ./data/webis/webis_dataset ./data/webis/webis_dataset_with_qualities
#python add_qualities_to_dataset.py ./data/twitter_100m/100m_tweets_dataset ./data/twitter_100m/100m_tweets_dataset_with_qualities
#python add_qualities_to_dataset.py ./data/wikipedia/wikipedia_dataset ./data/wikipedia/wikipedia_dataset_with_qualities $SLURM_ARRAY_TASK_ID
python add_qualities_to_dataset_parallel.py ./data/wikipedia/wikipedia_dataset ./data/wikipedia/wikipedia_dataset_with_qualities $SLURM_ARRAY_TASK_ID


# when all done, launch
#python merge_parallel_datasets.py ./data/wikipedia/wikipedia_dataset_with_qualities