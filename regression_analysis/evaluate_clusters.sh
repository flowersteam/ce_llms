#!/bin/bash
#SBATCH -A imi@h100
#SBATCH -C h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:59:59
#SBATCH --array=0-119
#SBATCH --cpus-per-task=24
#SBATCH -o logs/evaluate_clusters_%A_%a.log
#SBATCH -e logs/evaluate_clusters_%A_%a.log
#SBATCH -J evaluate_clusters
##SBATCH --qos=qos_gpu_h100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
#module purge
module load arch/h100
module load python/3.12.2 || module load python/3.12.8 || module load python/3.12.7
conda activate cluster_312

clustering_paths=(
#  data/webis/webis_dataset_clusters/results/*
#  data/twitter_100m/100m_tweets_dataset_clusters/results/*
#  data/reddit_submissions/reddit_submissions_dataset_clusters/results/*
  data/wikipedia/wikipedia_dataset_clusters/results/*
)

# Initialize an empty array to store paths that contain gen_19
echo "Number of paths: ${#clustering_paths[@]}"

# Parameters to iterate over
path=${clustering_paths[$SLURM_ARRAY_TASK_ID]}

#python -m regression_analysis.evaluate_clusters \
# --clustering-pkl $path \
# --dataset-path ./data/webis/webis_dataset_with_qualities \
# --embeddings-path ./data/webis/webis_dataset_embeddings.npy \
# --projections-path ./data/webis/webis_dataset_projections.npy \
# --clustering-evaluation-save-dir ./data/webis/webis_dataset_clusters/evaluation

#python -m regression_analysis.evaluate_clusters \
# --clustering-pkl $path \
# --dataset-path ./data/twitter_100m/100m_tweets_dataset_with_qualities \
# --embeddings-path ./data/twitter_100m/100m_tweets_dataset_embeddings.npy \
# --projections-path ./data/twitter_100m/100m_tweets_dataset_projections.npy \
# --clustering-evaluation-save-dir ./data/twitter_100m/100m_tweets_dataset_clusters/evaluation

#python -m regression_analysis.evaluate_clusters \
#--clustering-pkl $path \
#--dataset-path ./data/reddit_submissions/reddit_submissions_dataset_with_qualities \
#--embeddings-path ./data/reddit_submissions/reddit_submissions_dataset_embeddings.npy \
#--projections-path ./data/reddit_submissions/reddit_submissions_dataset_projections.npy \
#--clustering-evaluation-save-dir ./data/reddit_submissions/reddit_submissions_dataset_clusters/evaluation

python -m regression_analysis.evaluate_clusters \
--clustering-pkl $path \
--dataset-path ./data/wikipedia/wikipedia_dataset_with_qualities \
--embeddings-path ./data/wikipedia/wikipedia_dataset_embeddings.npy \
--projections-path ./data/wikipedia/wikipedia_dataset_projections.npy \
--clustering-evaluation-save-dir ./data/wikipedia/wikipedia_dataset_clusters/evaluation
