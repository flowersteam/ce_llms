#!/bin/bash
#SBATCH -A imi@cpu
#SBATCH --time=01:59:59
##SBATCH --array=0-0
#SBATCH --cpus-per-task=48
#SBATCH -o logs/run_on_node_cpu_log_%A.log
#SBATCH -e logs/run_on_node_cpu_log_%A.log
#SBATCH -J run_on_node_cpu
#SBATCH --qos=qos_cpu-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load python/3.12.2 || module load python/3.12.8 || module load python/3.12.7
conda activate cluster_312


##webis
#python -m regression_analysis.create_evaluation_umap_projections \
#    --embeddings-path ./data/webis/webis_dataset_embeddings.npy \
#    --umap-save-path ./data/webis/webis_dataset_umap.pkl \
#    --projections-save-path ./data/webis/webis_dataset_projections.npy
#
##100m_tweets
#python -m regression_analysis.create_evaluation_umap_projections \
#    --embeddings-path ./data/twitter_100m/100m_tweets_dataset_embeddings.npy \
#    --umap-save-path ./data/twitter_100m/100m_tweets_dataset_umap.pkl \
#    --projections-save-path ./data/twitter_100m/100m_tweets_dataset_projections.npy
#
##reddit_submissions
#python -m regression_analysis.create_evaluation_umap_projections \
#    --embeddings-path ./data/reddit_submissions/reddit_submissions_dataset_embeddings.npy \
#    --umap-save-path ./data/reddit_submissions/reddit_submissions_dataset_umap.pkl \
#    --projections-save-path ./data/reddit_submissions/reddit_submissions_dataset_projections.npy


#wikipedia
python -m regression_analysis.create_evaluation_umap_projections \
    --embeddings-path ./data/wikipedia/wikipedia_dataset_embeddings.npy \
    --umap-save-path ./data/wikipedia/wikipedia_dataset_umap.pkl \
    --projections-save-path ./data/wikipedia/wikipedia_dataset_projections.npy
