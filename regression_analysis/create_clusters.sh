#!/bin/bash
#SBATCH -A imi@cpu
#SBATCH --time=01:59:59
#SBATCH --array=0-125
#SBATCH --cpus-per-task=48
#SBATCH -o logs/create_clusters_%A_%a.log
#SBATCH -e logs/create_clusters_%A_%a.log
#SBATCH -J create_clusters

### webis:
## input data
#dataset_path="./data/webis/webis_dataset_with_qualities"
#embeddings_path="./data/webis/webis_dataset_embeddings.npy"
## output data
#clusters_save_dir="./data/webis/webis_dataset_clusters"
#visualization_save_dir="./viz_results/webis/clusters"

### 100m_tweets
## input data
#dataset_path="./data/twitter_100m/100m_tweets_dataset_with_qualities"
#embeddings_path="./data/twitter_100m/100m_tweets_dataset_embeddings.npy"
## output data
#clusters_save_dir="./data/twitter_100m/100m_tweets_dataset_clusters"
#visualization_save_dir="./viz_results/twitter_100m/clusters"

## reddit_submissions
## input data
#dataset_path="./data/reddit_submissions/reddit_submissions_dataset_with_qualities"
#embeddings_path="./data/reddit_submissions/reddit_submissions_dataset_embeddings.npy"
## output data
#clusters_save_dir="./data/reddit_submissions/reddit_submissions_dataset_clusters"
#visualization_save_dir="./viz_results/reddit_submissions/clusters"

## wikipedia
# input data
dataset_path="./data/wikipedia/wikipedia_dataset_with_qualities"
embeddings_path="./data/wikipedia/wikipedia_dataset_embeddings.npy"
# output data
clusters_save_dir="./data/wikipedia/wikipedia_dataset_clusters"
visualization_save_dir="./viz_results/wikipedia/clusters"

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
#module load arch/h100
module load python/3.12.2 || module load python/3.12.8 || module load python/3.12.7
conda activate cluster_312

# Parameters to iterate over
umap_min_dist=(0.0001 0.001 0.01)
noise=("" "--no-noise")

# Base command list (without min-dist and noise settings)
base_commands=(
    "--clustering-method 'gmm' --gmm-n-components 5"
    "--clustering-method 'gmm' --gmm-n-components 10"
    "--clustering-method 'gmm' --gmm-n-components 25"
    "--clustering-method 'gmm' --gmm-n-components 50"
    "--clustering-method 'gmm' --gmm-n-components 100"
    "--clustering-method 'gmm' --gmm-n-components 1000"
    "--clustering-method 'kmeans' --kmeans-n-clusters 5"
    "--clustering-method 'kmeans' --kmeans-n-clusters 10"
    "--clustering-method 'kmeans' --kmeans-n-clusters 25"
    "--clustering-method 'kmeans' --kmeans-n-clusters 50"
    "--clustering-method 'kmeans' --kmeans-n-clusters 100"
    "--clustering-method 'kmeans' --kmeans-n-clusters 1000"
    "--clustering-method 'dbscan' --dbscan-eps 0.1 --dbscan-min-samples 50"
    "--clustering-method 'dbscan' --dbscan-eps 0.2 --dbscan-min-samples 100"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 5 --hdbscan-min-samples 1 --hdbscan-cluster-selection-method 'eom'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 100 --hdbscan-min-samples 1 --hdbscan-cluster-selection-method 'leaf'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 1000 --hdbscan-min-samples 100 --hdbscan-cluster-selection-method 'leaf'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 1000 --hdbscan-min-samples 1 --hdbscan-cluster-selection-method 'leaf'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 1000 --hdbscan-min-samples 100 --hdbscan-cluster-selection-method 'leaf'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 3000 --hdbscan-min-samples 1 --hdbscan-cluster-selection-method 'leaf'"
    "--clustering-method 'hdbscan' --hdbscan-min-cluster-size 3000 --hdbscan-min-samples 100 --hdbscan-cluster-selection-method 'leaf'"
)


# Generate all clustering method calls
tasks=()
for dist in "${umap_min_dist[@]}"; do
    for n in "${noise[@]}"; do
        for cmd in "${base_commands[@]}"; do
            tasks+=("python -m regression_analysis.create_clusters \
                --dataset-path $dataset_path \
                --embeddings-path $embeddings_path \
                --clusters-save-dir $clusters_save_dir \
                --visualization-save-dir $visualization_save_dir \
                --min-focus-cluster-size 80000 \
                --plot-focus-umaps \
                --umap-n-neighbors 5 \
                --umap-min-dist $dist \
                $n \
                $cmd")
        done
    done
done

echo "Total number of tasks: ${#tasks[@]}"

# Execute task based on SLURM array ID
task=${tasks[$SLURM_ARRAY_TASK_ID]}
echo "Running task $SLURM_ARRAY_TASK_ID: $task"
eval $task