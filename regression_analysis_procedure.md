# Full analysis procedure

## Create conda env

```
conda create --name cluster_312 python=3.12
conda activate cluster_312
pip install -r requirements_cluster
```

## Add qualities to dataset
For this you need `vllm_312` conda env (see README.md).
```
sbatch add_qualities.sh
```

This launches the following command.
webis:
```
python add_qualities_to_dataset.py ./data/webis/webis_dataset ./data/webis/webis_dataset_with_qualities
```

100m_tweets: `python add_qualities_to_dataset.py ./data/twitter_100m/100m_tweets_dataset ./data/twitter_100m/100m_tweets_dataset_with_qualities`

## 1. embed your dataset

You can use `run_on_node_gpu.sh` script to run this on slurm cluster. (Modify it accordingly)

```
python regression_analysis/create_dataset_embeddings.py \
    --dataset_path ./data/webis/webis_dataset_with_qualities \
    --embeddings_path ./data/webis/webis_dataset_embeddings.npy
```

100m_tweets: `python regression_analysis/create_dataset_embeddings.py --dataset_path ./data/twitter_100m/100m_tweets_dataset_with_qualities --embeddings_path ./data/twitter_100m/100m_tweets_dataset_embeddings.npy`

reddit submissions: `python regression_analysis/create_dataset_embeddings.py --dataset_path ./data/reddit_submissions/reddit_submissions_dataset_with_qualities --embeddings_path ./data/reddit_submissions/reddit_submissions_dataset_embeddings.npy`


## 2. create clusters

### 2.1. Create clusters 
Set the parameters inside according to your dataset in `regression_analysis/create_clusters.sh`.

webis (1086 clusters are created, but can vary):
```
dataset_path="./data/webis/webis_dataset_with_qualities"
embeddings_path="./data/webis/webis_dataset_embeddings.npy"
clusters_save_dir="./data/webis/webis_dataset_clusters"
visualization_save_dir="./viz_results/webis/clusters"
```

Then launch 125 parallel clustering methods with slurm (regular machine: prepend `SLURM_ARRAY_TASK_ID=<number>`)
```
sbatch regression_analysis/create_clusters.sh
```

Each is a call with one clustering method, for instance :
```
python -m regression_analysis.create_clusters \
    --dataset-path ./data/webis/webis_dataset_with_qualities \
    --embeddings-path ./data/webis/webis_dataset_embeddings.npy \
    --clusters-save-dir ./data/webis/webis_dataset_clusters \
    --visualization-save-dir $visualization_dir ./viz_results/webis/clusters \
    ... clsutering parameters ..****.
```

This saves the clusters as datasets in `./data/webis/webis_dataset_clusters/umap_(..,..)_<method>_(..,..)_cluster_<cluster_label>`,
for instance `./data/webis/webis_dataset_clusters/umap_(5,0.0001)_gmm_(5)_cluster_3`

This also creates a results files in `./data/webis/webis_dataset_clusters/results/umap_(..,..)_<method>_(..,..).pkl`,
for instance `./data/webis/webis_dataset_clusters/results/umap_(5,0.0001)_gmm_(5).pkl`
This file contains logs and mappings of cluster_label -> cluster_path, cluster_indices (indices of elements in the full dataset)

All clusters are drawn in `./viz_results/webis/clusters`.


### 2.2. Saving and recreating clusters

If your cluster cleans files, be sure to transfer the results folder to a safe partitions.
Clusters can be recreated from the results folder, example :
```
python -m regression_analysis.recreate_clusters_from_results \
    --results-path "./data/webis/webis_dataset_clusters/results/umap_(5,0.0001)_gmm_(5).pkl" \
    --dataset-path ./data/webis/webis_dataset_with_qualities
```
Or you can do it for all results with
```
for results_file in ./data/webis/webis_dataset_clusters/results/*.pkl; do
    python -m regression_analysis.recreate_clusters_from_results \
        --results-path "$results_file" \
        --dataset-path ./data/webis/webis_dataset_with_qualities
done
```



## 3. Evaluate clusters

### 3.1. Umap projections
First create the UMAP projections for evaluation.

You can do this with sbatch `run_on_node_cpu.sh` on slurm machine, but set the call according to your dataset.

webis:
```
python -m regression_analysis.create_evaluation_umap_projections \
    --embeddings-path ./data/webis/webis_dataset_embeddings.npy \
    --umap-save-path ./data/webis/webis_dataset_umap.pkl \
    --projections-save-path ./data/webis/webis_dataset_projections.npy 
```

We want to use the same UMAP projection for evaluating all clusters for consistency.
This is different to the UMAPs that were fit for clustering. Those are different for each clustering method to increase diversity.

Transfer *umap.pkl and *dataset_projections.npy to safe partition.


### 3.2. Evaluate clusters using the precomputed umap projections

Launches 120 parallel evaluations of clusters. 

Set parameters in `regression_analysis/evaluate_clusters.sh`:

- set `clustering_paths` and the arguments for evaluate_clusters
- set call

webis:
```
python -m regression_analysis.evaluate_clusters \
 --clustering-pkl "data/webis/webis_dataset_clusters/results/umap_(5,0.0001)_gmm_(5).pkl" \
 --dataset-path ./data/webis/webis_dataset_with_qualities \
 --embeddings-path ./data/webis/webis_dataset_embeddings.npy \
 --projections-path ./data/webis/webis_dataset_projections.npy \
 --clustering-evaluation-save-dir ./data/webis/webis_dataset_clusters/evaluation
```


Then run (slurm):
```
sbatch regression_analysis/evaluate_clusters.sh
```

This creates results for each cluster in
`./data/webis/webis_dataset_clusters/evaluation/umap_(..,..)_<method>_(..,..)_cluster_<cluster_label>.pkl`.
(Transfer this to safe partition)

# 4. Select representative clusters

webis:
```
python -m regression_analysis.select_clusters \
    --cluster-evaluation-files ./data/webis/webis_dataset_clusters/evaluation/*.pkl \
    --visualization-dir viz_results/webis/clusters \
    --selection-save-path ./data/webis/selected_clusters_indices_to_path.json
```
100m_tweets:
```
python -m regression_analysis.select_clusters \
    --cluster-evaluation-files ./data/twitter_100m/100m_tweets_dataset_clusters/evaluation/*.pkl \
    --visualization-dir viz_results/twitter_100m/clusters \
    --selection-save-path ./data/twitter_100m/selected_clusters_indices_to_path.json
```

100m_tweets:
```
python -m regression_analysis.select_clusters \
    --cluster-evaluation-files ./data/reddit_submissions/reddit_submissions_dataset_clusters/evaluation/*.pkl \
    --visualization-dir viz_results/reddit_submissions/clusters \
    --selection-save-path ./data/reddit_submissions/selected_clusters_indices_to_path.json
```

wikipedia:
```
python -m regression_analysis.select_clusters \
    --cluster-evaluation-files ./data/wikipedia/wikipedia_dataset_clusters/evaluation/*.pkl \
    --visualization-dir viz_results/wikipedia/clusters \
    --selection-save-path ./data/wikipedia/selected_clusters_indices_to_path.json
```

This creates the following five files (example 100m_tweets):
`viz_results/twitter_100m/clusters/metrics_distributions.png`
`viz_results/twitter_100m/clusters/clustering_metrics_distributions.pkl`
`viz_results/twitter_100m/clusters/selected_clusters_correlations.png`
`data/twitter_100m/selected_clusters_indices_to_path.json`

Then transfer them selection to a safe partition.


# 5. Launch simulations and evaluate

##  5.1 Launch simulations and evaluate all except quality
Make sure you have `unsloth_312`, `eval_312`, and `vllm_312` conda envs (see README.md).

Set your arguments in `clusters_iterative_train.sh`. 
You have to set the `dataset_name` variable and the `ratios`. (also check that cluster loading is implemented in `dataset_utils.py`)

This launches 400 simulations on a slurm cluster.
```
sbatch clusters_iterative_train.sh 
```
Same can be done on a regular machine by e.g. `SLURM_ARRAY_TASK_ID=0 bash clusters_iterative_train.sh` to launch the first simulation.

This will simulate iterative chains and also evaluate all metrics except quality.

webis:
You should see the simulation results in `simulation_results/webis_clusters/`
and the evaluation results in `eval_results/simulation_results/webis_clusters/`.

100m_tweets:
simulation results: `simulation_results/100m_tweets_clusters/`
evaluation results: `eval_results/simulation_results/100m_tweets_clusters/`.


##  5.2 Evaluate quality
To evaluate the quality you need `vllm_312` conda env (see README)

To do that the simplest is to set the paths in `run_vllm_eval.sh` and then run `sbatch run_vllm_eval.sh`
Or you can run it with bash on a regular machine (`bash run_vllm_eval.sh`)

This will run the following command on all the selected paths:
```
evaluate_generations.py --llama-quality-scale ....
```
This will append the quality scores to the results in e.g. `eval_results/simulation_results/webis_clusters/`

# 6. Visualize results (not needed for clusters)
See generate_plots.sh for example usages
```
python show_sample_generations.py --experiment-dir results/Testing_iterative_learning_instructions_deduplicate_n_4000_temp_0.7/ 
python visualize.py --metric div --directories eval_results/Testing_iterative_learning_*/part_*
```

loads from: "./data/webis/selected_clusters_indices_to_path.json"
save u webis_clusters_results_v2


# 7. Regression analyses
You can see examples of running regression analyses in
```
regression_analysis/run_regression_analyses.sh
```
For instance,
```
python -m regression_analysis.regression_analysis d -g 500 --no-show --cv 10 \
    --save-dir regression_analysis/results/webis_clusters \
    --clusters-indices-to-path-json data/webis/selected_clusters_indices_to_path.json \
    --clusters-evaluation-dir data/webis/webis_dataset_clusters/evaluation \
    --clusters-simulation-results-dir eval_results/simulation_results/webis_clusters
```
This runs the analysis and saves the figures in `regression_analysis/results/webis_clusters/`