plots_dir="./viz_results/collapse_plots_v2"
mkdir -p $plots_dir/"q_abs"
mkdir -p $plots_dir/"d_abs"
mkdir -p $plots_dir/"q_pc"
mkdir -p $plots_dir/"d_pc"

metrics=(
#      'text_len_cap_10000'
#      'ttr_cap_10000'
#      'n_unique_words_total_cap_10000'
#      'llama_quality_scale'
#      'word_entropy_cap_10000'
#      'diversity_selfbleu_cap_10000'
#      'cos_diversity_cap_10000'
#       'knn_5_cos_diversity_cap_250'
#       'knn_50_cos_diversity_cap_250'
#       'knn_5_cos_diversity_cap_10000'
#       'knn_50_cos_diversity_cap_10000'
#      'knn_100_cos_diversity_cap_10000'
#       'knn_1000_cos_diversity_cap_10000'
#       'kl_entropy_cap_250_k_5'
#       'kl_entropy_cap_250_k_50'
#       'kl_entropy_cap_10000_k_5'
#      'kl_entropy_cap_10000_k_50'
#       'kl_entropy_cap_10000_k_100'
#       'kl_entropy_cap_10000_k_1000'
#      'kl_entropy_cap_10000_k_2000'
#      'gaussian_aic_cap_10000'
#       'toxicity_cap_250'
       'toxicity_cap_10000'
#       'positivity_cap_250'
       'positivity_cap_10000'
#       'aggregate_reading_level_cap_250'
       'aggregate_reading_level_cap_10000'
)

for m in "${metrics[@]}"; do

  dirs_pattern=""
  dirs_pattern+=" eval_results/webis_clusters_results_v2/*webis*type_cluster*_1/generated_*_*"
#  python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --predictor-metric "cluster_"$m --no-show --save-path ${plots_dir}/q_abs/$m
#  python plot_collapse.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --predictor-metric "cluster_"$m --no-show --save-path ${plots_dir}/d_abs/$m

  # norm
  python plot_collapse.py --directories $dirs_pattern -n --metric cos_diversity_stella_cap_250 --predictor-metric "cluster_"$m --no-show --save-path ${plots_dir}/d_pc/$m
  python plot_collapse.py --directories $dirs_pattern -n --metric llama_quality_scale_cap_250 --predictor-metric "cluster_"$m --no-show --save-path ${plots_dir}/q_pc/$m

done