#!/bin/bash
##SBATCH -A imi@h100
##SBATCH -C h100
##SBATCH -A vgw@a100
##SBATCH -C a100
##SBATCH --gres=gpu:4
##SBATCH --gres=gpu:1
#SBATCH -A imi@cpu
##SBATCH --time=00:59:59
#SBATCH --time=01:59:59
#SBATCH --array=0-125
#SBATCH --cpus-per-task=48
#SBATCH -o logs/run_miniclusters_%A_%a.log
#SBATCH -e logs/run_miniclusters_%A_%a.log
#SBATCH -J run_on_node
##SBATCH --qos=qos_gpu_h100-dev
##SBATCH --qos=qos_gpu_a100-dev

source /linkhome/rech/genini01/utu57ed/.bashrc
module purge
module load arch/h100
module load python/3.11.5
#conda activate eval_311
conda activate clustering_311

# Parameters to iterate over
umap_min_dist=(0.0001 0.001 0.01)
noise=("" "--no_noise")

# Base command list (without min_dist and noise settings)
base_commands=(
    "--clustering_method 'gmm' --gmm_n_components 5"
    "--clustering_method 'gmm' --gmm_n_components 10"
    "--clustering_method 'gmm' --gmm_n_components 25"
    "--clustering_method 'gmm' --gmm_n_components 50"
    "--clustering_method 'gmm' --gmm_n_components 100"
    "--clustering_method 'gmm' --gmm_n_components 1000"
    "--clustering_method 'kmeans' --kmeans_n_clusters 5"
    "--clustering_method 'kmeans' --kmeans_n_clusters 10"
    "--clustering_method 'kmeans' --kmeans_n_clusters 25"
    "--clustering_method 'kmeans' --kmeans_n_clusters 50"
    "--clustering_method 'kmeans' --kmeans_n_clusters 100"
    "--clustering_method 'kmeans' --kmeans_n_clusters 1000"
    "--clustering_method 'dbscan' --dbscan_eps 0.1 --dbscan_min_samples 50"
    "--clustering_method 'dbscan' --dbscan_eps 0.2 --dbscan_min_samples 100"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 5 --hdbscan_min_samples 1 --hdbscan_cluster_selection_method 'eom'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 100 --hdbscan_min_samples 1 --hdbscan_cluster_selection_method 'leaf'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 1000 --hdbscan_min_samples 100 --hdbscan_cluster_selection_method 'leaf'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 1000 --hdbscan_min_samples 1 --hdbscan_cluster_selection_method 'leaf'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 1000 --hdbscan_min_samples 100 --hdbscan_cluster_selection_method 'leaf'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 3000 --hdbscan_min_samples 1 --hdbscan_cluster_selection_method 'leaf'"
    "--clustering_method 'hdbscan' --hdbscan_min_cluster_size 3000 --hdbscan_min_samples 100 --hdbscan_cluster_selection_method 'leaf'"
)

#                --use_cache \
# Generate all combinations
tasks=()
for dist in "${umap_min_dist[@]}"; do
    for n in "${noise[@]}"; do
        for cmd in "${base_commands[@]}"; do
            tasks+=("python create_webis_miniclusters.py \
                --exp_tag 'miniclusters_merge_80k' \
                --min_focus_cluster_size 80000 \
                --plot_focus_umaps \
                --umap_n_neighbors 5 \
                --umap_min_dist $dist \
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