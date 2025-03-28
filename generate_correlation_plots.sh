plots_dir="./viz_results/experiments_new"
mkdir -p  $plots_dir"/arrows"
dirs_pattern=""
dirs_pattern+=" eval_results/webis_clusters_results_v2/*webis*type_cluster*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*100m*type_st*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*senator_t*type_st*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*t_submissions*type_st*_1/generated_*_*"
#dirs_pattern+=" eval_results/quality_results/*webis*type_st*_1/generated_*_*"
#python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric gaussian_aic_cap_80000
python correlate_metrics.py --arrows -hm --directories $dirs_pattern --metrics llama_quality_scale_cap_250 cos_diversity_stella_cap_250 #--no-show --save-path ${plots_dir}/arrows/Qselfbleu_abs
exit
#python correlate_metrics.py --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/arrows/QD_abs
#python correlate_metrics.py -n --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/arrows/QD_pc
python correlate_metrics.py --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 diversity_selfbleu_cap_250 --no-show --save-path ${plots_dir}/arrows/Qselfbleu_abs
python correlate_metrics.py -n --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 diversity_selfbleu_cap_250 --no-show --save-path ${plots_dir}/arrows/Qselfbleu_pc
exit

#dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
#python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_15_nn_spread_stella_cap_250
dirs_pattern+=" eval_results/quality_results/*webis*type_cluster_v2*_1/generated_*_*"
#python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_5_nn_spread_stella_cap_250
#exit

plots_dir="./viz_results/experiments_new"
mkdir -p  $plots_dir"/collapse"
dirs_pattern=" eval_results/quality_results/*senator_t*type_Q*_1/generated_*_*"
python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_15_nn_spread_stella_cap_250 --no-show --save-path ${plots_dir}/collapse/senator_collapse
dirs_pattern=" eval_results/quality_results/*100m*type_Q*_1/generated_*_*"
python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_15_nn_spread_stella_cap_250 --no-show --save-path ${plots_dir}/collapse/100M_collapse
dirs_pattern=" eval_results/quality_results/*reddit_submissions*type_Q*_1/generated_*_*"
python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_15_nn_spread_stella_cap_250 --no-show --save-path ${plots_dir}/collapse/submissions_collapse
dirs_pattern=" eval_results/quality_results/*webis*type_Q*_1/generated_*_*"
python plot_collapse.py --directories $dirs_pattern --metric llama_quality_scale_cap_250 --dep-metric cos_15_nn_spread_stella_cap_250 --no-show --save-path ${plots_dir}/collapse/webis_collapse
#python correlate_metrics.py --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 cos_diversity_stella_cap_250
#python correlate_metrics.py --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 diversity_selfbleu_cap_250
exit

plots_dir="./viz_results/experiments_draft"
mkdir -p $plots_dir

### Correlations
################
mkdir -p "./viz_results/experiments_draft/llama_five"
# llama_five
dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --arrows --directories $dirs_pattern --metrics llama_quality_five_cap_250 cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/llama_five/qd_arrows
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_all_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_all_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_all_llama_gib

dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_other_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_other_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_other_llama_gib

dirs_pattern=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_100m_tw_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_100m_tw_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_five_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_five/corr_100m_tw_llama_gib

# llama_scale
mkdir -p "./viz_results/experiments_draft/llama_scale"

dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --arrows --directories $dirs_pattern --metrics llama_quality_scale_cap_250 cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/llama_scale/qd_arrows
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_all_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_all_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_all_llama_gib

dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_other_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_other_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_other_llama_gib

dirs_pattern=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_100m_tw_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_100m_tw_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_scale_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_scale/corr_100m_tw_llama_gib

# llama_gibberish
mkdir -p "./viz_results/experiments_draft/llama_gibberish"
dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --arrows --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/qd_arrows
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_all_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_all_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_all_llama_gib

dirs_pattern=" eval_results/results/human_ai*rstrip*/gen*" # webis reddit
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*" # reddit submissions
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_senator_tweets_type_standard*_1/generated_*_*" # senator tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_other_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_other_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_other_llama_gib

dirs_pattern=" eval_results/results/human_ai_ratio_dataset_100m_tweets_cl_type_standard*_1/generated_*_*" # 100M tweets
python correlate_metrics.py  --directories $dirs_pattern --metrics aggregate_reading_level_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_100m_tw_reading_lvl_gib
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 aggregate_reading_level_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_100m_tw_llama_reading_lvl
python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_gibberish_cap_250 gibberish_score_cap_250 --no-show --save-path ${plots_dir}/llama_gibberish/corr_100m_tw_llama_gib

