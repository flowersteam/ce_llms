#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:4
#SBATCH --time=01:59:59
##SBATCH --array=0-0
#SBATCH --cpus-per-task=24
#SBATCH -o logs/run_on_node_gpu_log_%A.log
#SBATCH -e logs/run_on_node_gpu_log_%A.log
#SBATCH -J run_on_node_gpu
#SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load python/3.12.2 || module load python/3.12.8 || module load python/3.12.7
conda activate cluster_312


#python regression_analysis/create_dataset_embeddings.py \
#    --dataset_path ./data/webis/webis_dataset \
#    --embeddings_path ./data/webis/webis_dataset_embeddings.npy



#python regression_analysis/create_dataset_embeddings.py \
#    --dataset_path ./data/twitter_100m/100m_tweets_dataset_with_qualities \
#    --embeddings_path ./data/twitter_100m/100m_tweets_dataset_embeddings.npy

#python regression_analysis/create_dataset_embeddings.py \
#    --dataset_path ./data/reddit_submissions/reddit_submissions_dataset_with_qualities \
#    --embeddings_path ./data/reddit_submissions/reddit_submissions_dataset_embeddings.npy

python regression_analysis/create_dataset_embeddings.py \
    --dataset_path ./data/wikipedia/wikipedia_dataset \
    --embeddings_path ./data/wikipedia/wikipedia_dataset_embeddings.npy
