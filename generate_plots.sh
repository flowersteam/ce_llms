#plots_dir="./viz_results/"
#mkdir -p $plots_dir"/merged"
#dirs_pattern=""
#
## Len exp
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_*/generated_*_*"
#
## Merged
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_wikipedia/generated_1000_*"
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_100m_tweets/generated_1000_*"
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_webis_reddit/generated_1000_*"
#
##dirs_pattern+=" eval_results/quality_results/*wikipedia*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*100m*type_standard*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_standard*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*reddit_submissions*type_standard*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*senator_tweets*type_standard*_1/generated_*_*"
#
## individual
#dirs_pattern+=" eval_results/simulation_results/human_ai_ratio_*100m*type_standard*_1/generated_*_*"
#dirs_pattern+=" eval_results/simulation_results/human_ai_ratio_*webis*type_standard*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*wikipedia*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_standard*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_standard*_1/generated_*_*"
#
## clusters
##dirs_pattern+=" eval_results/simulation_results/wikipedia_clusters/clusters_dataset_wikipedia*/generated_*_*"
#
##dirs_pattern+=" eval_results/quality_results/*100m*type_short*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*100m*type_medium*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*100m*type_long*_1/generated_*_*"
#
###dirs_pattern+=" eval_results/quality_results/*webis*type_short*_1/generated_*_*"
###dirs_pattern+=" eval_results/quality_results/*webis*type_medium*_1/generated_*_*"
###dirs_pattern+=" eval_results/quality_results/*webis*type_long*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*Q*_1/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
##dirs_pattern+=" eval_results/webis_clusters_results_v2/*webis*type_cluster*_1/generated_*_*"
##dirs_pattern+=" eval_results/simulation_results/webis_clusters/clusters_dataset_webis_*/generated_1000_*"
#
##python visualize.py  --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter
##python visualize.py  --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter
##exit
##
##python visualize.py  --legend-conf merged --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/abs_individual_quality --assert-n-datapoints 5
##python visualize.py  --legend-conf merged --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/abs_individual_diversity --assert-n-datapoints 5
##python visualize.py -n --legend-conf merged --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/rel_individual_quality --assert-n-datapoints 5
##python visualize.py -n --legend-conf merged --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/rel_individual_diversity --assert-n-datapoints 5
##exit
#
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_*/generated_*_*"
#
##dirs_pattern=""
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_wikipedia/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_100m_tweets/generated_*_*"
##dirs_pattern+=" eval_results/quality_results/*merged*type_standard*_1_partition_webis_reddit/generated_*_*"
#
##python visualize.py  --legend-conf merged --directories $dirs_pattern --metric llama_quality_scale_cap_80 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_abs --assert-n-datapoints 5
##python visualize.py  --legend-conf merged --directories $dirs_pattern --metric cos_diversity_stella_cap_80 --scatter  #--no-show --save-path ${plots_dir}/na_cluster/interaction_plots_bleu_abs --assert-n-datapoints 5
#
#python visualize.py --legend-conf merged  --directories $dirs_pattern --metric llama_quality_scale_cap_80 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/abs_merged_quality --assert-n-datapoints 5
#python visualize.py --legend-conf merged  --directories $dirs_pattern --metric cos_diversity_stella_cap_80 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/abs_merged_diversity --assert-n-datapoints 5
#python visualize.py -n --legend-conf merged  --directories $dirs_pattern --metric llama_quality_scale_cap_80 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/rel_merged_quality --assert-n-datapoints 5
#python visualize.py -n --legend-conf merged  --directories $dirs_pattern --metric cos_diversity_stella_cap_80 -ip -ig 19 -igs 5 -fhr 1.0 --scatter  --no-show --save-path ${plots_dir}/merged/rel_merged_diversity --assert-n-datapoints 5
#
#for path in viz_results/merged/*pdf; do convert "$path" "${path%.pdf}.png"; done
#
#exit
#
##dirs_pattern+=" eval_results/quality_results/*100*type_s*_1/generated_*_*"
#
#
plots_dir="./viz_results/experiments_new"
#mkdir -p $plots_dir
#
#
## 1.s evolution (webis)
#############
#mkdir -p $plots_dir"/evo"
#dirs_pattern=" eval_results/quality_results/*100m*standard*_1/generated_*_*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 -igs 5 --scatter --no-show --save-path ${plots_dir}/evo/evolution_100m_quality_abs --assert-n-datapoints 5
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -igs 5 --scatter --no-show --save-path ${plots_dir}/evo/evolution_100m_diversity_abs --assert-n-datapoints 5

# Interaction plots
############
mkdir -p $plots_dir"/int"
dirs_pattern=""
dirs_pattern+=" eval_results/quality_results/*webis*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*100m*standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*reddit_submissions*standard*_1/generated_*_*"

dirs_pattern+=" eval_results/quality_results/*senator_tw*standard*_1/generated_*_*"
#dirs_pattern+=" eval_results/simulation_results/human_ai_ratio_*100m*type_standard*_1/generated_*_*"
#dirs_pattern+=" eval_results/simulation_results/human_ai_ratio_*webis*type_standard*_1/generated_*_*"
dirs_pattern+=" eval_results/quality_results/*wikipedia*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_standard*_1/generated_*_*"

python visualize.py --no-legend --legend-conf datasets -n --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_quality_pc --assert-n-datapoints 5
python visualize.py --no-legend --legend-conf datasets --directories $dirs_pattern --metric llama_quality_scale_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_quality_abs --assert-n-datapoints 5
python visualize.py --no-legend --legend-conf datasets -n --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_diversity_pc --assert-n-datapoints 5
python visualize.py --no-legend --legend-conf datasets --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --no-show --save-path ${plots_dir}/int/interaction_plots_diversity_abs --assert-n-datapoints 5
exit