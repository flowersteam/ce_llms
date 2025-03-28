#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --time=01:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o logs/vllm_%A.client.log
#SBATCH -e logs/vllm_%A.client.log
#SBATCH -J vllm_server
#SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.10.4
#conda activate vllm
conda activate vllm_311

# for llama first launch vllm server with

#export VLLM_CONFIGURE_LOGGING=0 # no logging
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SERVERLOG="logs/vllm_$SLURM_JOB_ID.server.log"
echo "Server log : $SERVERLOG"

vllm serve /lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/ \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.98 \
    --served-model-name llama \
    --dtype bfloat16 \
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
echo "Launching client"
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311


#python prepare_senator_tweets_dataset.py
#python prepare_reddit_submissions_dataset.py
#python prepare_100m_tweets_dataset.py
#python prepare_webis_dataset.py
python play_with_dataset.py


# then wait for the model to load (tail -f logs/vllm_server_logs...)
# connect to same note with ssh
# launch run_localy.sh