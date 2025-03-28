#dirs_pattern=" eval_results/quality_results/*senator*Q*_1/generated_*_*"
#dirs_pattern=" eval_results/quality_results/*100m*Q*_1/generated_*_*"
#python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5
#exit
#plots_dir="./viz_results/experiments_new"
#mkdir -p $plots_dir"/na_cluster"
#dirs_pattern=""
# Len exp
#dirs_pattern+=" eval_results/quality_results/*senator*type_short*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*senator*type_long*_1/generated_*_*"

#dirs_pattern+=" eval_results/quality_results/*100m*type_short*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_medium*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_long*_1/generated_*_*"

##dirs_pattern+=" eval_results/quality_results/*webis*type_short*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_medium*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_long*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*webis*Q*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
#dirs_pattern+=" eval_results/webis_clusters_results_v2/*webis*type_cluster*_1/generated_*_*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 0.25 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_diversity_abs --assert-n-datapoints 5
python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 0.25 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --assert-n-datapoints 5 ${plots_dir}/na_cluster/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_pc --assert-n-datapoints 5
exit

#dirs_pattern+=" eval_results/quality_results/*100*type_s*_1/generated_*_*"


plots_dir="./viz_results/experiments_new"
mkdir -p $plots_dir


# 1.s evolution (webis)
############
mkdir -p $plots_dir"/evo"
dirs_pattern=" eval_results/quality_results/*100m*standard*_1/generated_*_*"
python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -igs 5 --scatter --no-show --save-path ${plots_dir}/evo/evolution_100m_quality_abs --assert-n-datapoints 5
python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -igs 5 --scatter --no-show --save-path ${plots_dir}/evo/evolution_100m_diversity_abs --assert-n-datapoints 5

# 1.b interaction plots
############
mkdir -p $plots_dir"/int"
dirs_pattern=""
dirs_pattern+=" eval_results/quality_results/*webis*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*100m*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*senator_tw*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*reddit_submissions*standard*_1/generated_*_*"
python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_quality_pc --assert-n-datapoints 5
python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_quality_abs --assert-n-datapoints 5
python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_diversity_pc --assert-n-datapoints 5
python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_bleu_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_bleu_abs --assert-n-datapoints 5
#exit


# Scaling
############
dirs_pattern=" eval_results/results/scale_unbalanced_sampling_small_mixed_dataset_webis_reddit_type_standard_presampled_split_all_acc_1_ft_size_4000_mixed_participants_*/gen*"
#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_diversity --assert-n-datapoints 5
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_quality --assert-n-datapoints 5
#
## scaling 1/10
dirs_pattern=" eval_results/results/scale_small_4k_train_mixed_dataset_webis_reddit_type_standard_presampled_split_all_acc_1_ft_size_4000_gen_train_ratio_0.1_participants_*/gen*"
#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_10pc_diversity --assert-n-datapoints 5
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_10pc_quality # --assert-n-datapoints 5 # 15 Nans (one seed less)

# Clusters
############
plots_dir="./viz_results/experiments_new"
mkdir -p $plots_dir"/clusters"
dirs_pattern=""
dirs_pattern+=" eval_results/quality_results/*webis*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*webis*type_cluster*_1/generated_*_*"

#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_bleu_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters/interaction_plots_bleu_pc --assert-n-datapoints 5

# Clusters v2
############
plots_dir="./viz_results/experiments_new"
mkdir -p $plots_dir"/clusters_v2"
dirs_pattern=""
dirs_pattern+=" eval_results/quality_results/*webis*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_bleu_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/clusters_v2/interaction_plots_bleu_pc --assert-n-datapoints 5
#exit
# Length
############
plots_dir="./viz_results/experiments_new"
mkdir -p $plots_dir"/length"
dirs_pattern=""
dirs_pattern+=" eval_results/quality_results/*webis*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*webis*type_short*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*webis*type_medium*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*webis*type_long*_1/generated_*_*"

#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_bleu_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/length/interaction_plots_bleu_pc --assert-n-datapoints 5
#exit

# QUALITY
############
mkdir -p $plots_dir"/Q"
# 100M_tweets
dirs_pattern=" eval_results/quality_results/*100m*Q*_1/generated_*_*"
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_quality_scale_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_quality_scale_abs
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_diversity_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_diversity_abs
#python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_bleu_pc
#python visualize.py --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_100m_tweets_bleu_abs

# senator
dirs_pattern=" eval_results/quality_results/*senator*Q*_1/generated_*_*"
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_quality_scale_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_quality_scale_abs
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_diversity_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_diversity_abs
#python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_bleu_pc
#python visualize.py --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_senator_tweets_bleu_abs

# reddit submissions
dirs_pattern=" eval_results/quality_results/*submissions*Q*_1/generated_*_*"
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_quality_scale_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_quality_scale_abs
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_diveristy_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_diversity_abs
#python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_bleu_pc
#python visualize.py --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_reddit_submissions_bleu_abs

# webis
dirs_pattern=" eval_results/quality_results/*webis*Q*_1/generated_*_*"
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_quality_scale_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_quality_scale_abs
python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_diversity_pc
python visualize.py --legend-conf qd --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_diversity_abs
#python visualize.py -n --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_bleu_pc
#python visualize.py --legend-conf qd --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/Q/q_webis_reddit_bleu_abs
exit

## Merged
##########
#plots_dir="./viz_results/experiments_new"
#mkdir -p $plots_dir"/merged"
#
#dirs_pattern+=" eval_results/quality_results/*senator_t*type_s*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*reddit_submissions*type_s*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*senator_submissions*type_s*_1/generated_*_*"
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_diversity --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_bleu --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/merged/interaction_plots_bleu_abs --assert-n-datapoints 5
#exit

# Len exp
dirs_pattern+=" eval_results/quality_results/*senator*type_short*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*senator*type_long*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_short*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_medium*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_long*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_short*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_medium*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_long*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*webis*Q*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
#dirs_pattern+=" eval_results/webis_clusters_results_v2/*webis*type_cluster*_1/generated_*_*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 0.25 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_diversity_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 0.25 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_pc --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --assert-n-datapoints 5 ${plots_dir}/na_cluster/interaction_plots_diversity_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric diversity_selfbleu_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_pc --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_abs --assert-n-datapoints 5
#python visualize.py -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter # --no-show --save-path ${plots_dir}/na_cluster/interaction_plots_quality_pc --assert-n-datapoints 5
exit
