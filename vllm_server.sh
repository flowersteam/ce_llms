#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --time=9:59:59
#SBATCH --cpus-per-task=24
#SBATCH -o logs/vllm_client_log_%A.log
#SBATCH -e logs/vllm_client_log_%A.log
#SBATCH -J vllm_server
##SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.10.4
conda activate vllm

# for llama first launch vllm server with

export VLLM_CONFIGURE_LOGGING=0 # no logging
export VLLM_WORKER_MULTIPROC_METHOD=spawn

SERVERLOG="logs/vllm_server_log_$SLURM_JOB_ID"
echo "Server log : $SERVERLOG"

#CUDA_VISIBLE_DEVICES=0,1 vllm serve /lustre/fsn1/projects/rech/imi/utu57ed/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/05917295788658563fd7ef778b6240ad9867d6d1/ \
#    --max-model-len 2048 \
#    --gpu-memory-utilization 0.98 \
#    --served-model-name llama \
#    --dtype half \
#    --disable-log-requests \
#    --disable-log-stats \
#    --enable-prefix-caching \
#    --tensor-parallel-size 2 &

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
echo "Launching client"
module purge
module load arch/h100
module load python/3.11.5
conda activate eval_311


#CUDA_VISIBLE_DEVICES=2 python play_with_dataset.py
python prepare_reddit_submissions_dataset.py


# then wait for the model to load (tail -f logs/vllm_server_logs...)
# connect to same note with ssh
# launch run_localy.sh