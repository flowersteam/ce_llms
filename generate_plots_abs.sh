# TEMP
dirs_pattern=""
#dirs_pattern+=" eval_results/results/scale_small_mixed*_*/generated_*_*"
#dirs_pattern+=" eval_results/results/scale_unbalanced_sampling_small_mixed*_*/generated_*_*"
#dirs_pattern+="eval_results/results/scale_small_mixed_dataset_webis_reddit_type_standard_presampled_split_all_acc_1_ft_size_4000_gen_train_ratio_0.1_participants_*/generated_*_*"
dirs_pattern+="eval_results/results/scale_small_4k_train_mixed_dataset_webis_reddit_type_standard_presampled_split_all_acc_1_ft_size_4000_gen_train_ratio_0.1_participants_*/generated_*_*"

#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250
#python visualize.py --directories $dirs_pattern --metric pc_unique_posts_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0
python visualize.py --directories $dirs_pattern --metric gibberish_score_cap_250 -ip -ig 19 -igs 5 -fhr 1.0
#python plot_scaling_law.py --directories $dirs_pattern --metric gibberish_score_cap_250 --generation 19 --smooth-n 5 --assert-n-datapoints 5
#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 19 --smooth-n 5 --assert-n-datapoints 5
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 19 --smooth-n 5 --assert-n-datapoints 5

#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 19 --smooth-n 5
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 19 --smooth-n 5
#python plot_scaling_law.py --directories $dirs_pattern --metric gibberish_score_cap_250 --generation 19 --smooth-n 5
exit

#dirs_pattern=""
#dirs_pattern+=" eval_results/results/scale_unbalanced_sampling_small_mixed*_1/generated_*_*"
#dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_webis*_1/generated_*_*"
#dirs_pattern+=" eval_results/results/scale_unbalanced_sampling_small_mixed*_4/generated_*_*"
#python visualize.py --directories $dirs_pattern --metric toxicity_cap_250 -ip -ig 19 -igs 5 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 19 -igs 5
#exit




# iterative train
######################

plots_dir="./viz_results/chain_QD"
mkdir -p $plots_dir

## quality diversity
##############
# webis
dirs_pattern=""
dirs_pattern+=" eval_results/results/scale_unbalanced_sampling_small_mixed*_1/generated_*_*"
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_webis_reddit_type_[hm]q*_1/generated_*_*"
python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/webis_reddit_diversity
python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/webis_reddit_quality
#python visualize.py --directories $dirs_pattern --metric gibberish_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/webis_reddit_gibberish
exit

# twitter100m
dirs_pattern=""
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_100m_tweets_type_standard*_1/generated_*_*"
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_100m_tweets_type_[hm]q*_1/generated_*_*"
python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/100m_tweets_diversity
python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/100m_tweets_quality
#python visualize.py --directories $dirs_pattern --metric gibberish_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/100m_tweets_gibberish

# reddit submissions
dirs_pattern=""
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_standard*1/generated_*_*"
dirs_pattern+=" eval_results/results/human_ai_ratio_dataset_reddit_submissions_type_[hm]q*1/generated_*_*"
python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/reddit_submissions_diversity
python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/reddit_submissions_quality
#python visualize.py --directories $dirs_pattern --metric gibberish_quality_cap_250 -ip -ig 19 -igs 5 -fhr 1.0 --scatter --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/reddit_submissions_gibberish
exit

#python visualize.py  --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/QD_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/QD_quality


# POPULATION
##############
dirs_pattern=""
#dirs_pattern=" eval_results/dev_results_scale/scale_v2*/generated_*"
# except 250
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/generated_*000_hu*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_1_ft_size_4000_mixed_participants_2/generated_*000_hu*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_type_standard_split_test_acc_1_ft_size_4000_mixed_participants_4/generated_*00_hu*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/generated_500_hu*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_1_ft_size_4000_mixed_participants_2/generated_500_hu*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit_type_standard_split_test_acc_1_ft_size_4000_mixed_participants_4/generated_500_hu*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 4 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 9 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 14 -fhr 1.0
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 19 -fhr 1.0 -igs 5

dirs_pattern=" eval_results/results/scale_v3*/generated_*"
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_diverisity
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/scaling_interaction_quality
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/scaling_interaction_diversity
python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_law_quality
python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 19 --smooth-n 5 --no-show --save-path ${plots_dir}/scaling_law_diversity
exit
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_diverisity
#exit

#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 4 --kl
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 4 --kl
#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 4 --kl --no-show --save-path ${plots_dir}/population_scaling_webis_reddit_diverisity_KL
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 4 --kl --no-show --save-path ${plots_dir}/population_scaling_webis_reddit_quality_KL
#python plot_scaling_law.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --generation 4 --no-show --save-path ${plots_dir}/population_scaling_webis_reddit_diverisity
#python plot_scaling_law.py --directories $dirs_pattern --metric llama_quality_cap_250 --generation 4 --no-show --save-path ${plots_dir}/population_scaling_webis_reddit_quality
#exit

# ACCUMULATION
##############
#dirs_pattern="eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_*_ft_size_4000_mixed_participants_1/*"
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/accumulation_reddit_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/accumulation_reddit_quality

#dirs_pattern="eval_results/old_results/human_ai_ratio_dataset_senator_tweets_split_all_acc_*_ft_size_4000_mixed_participants_1/*"
#python visualize.py  --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/accumulation_twitter_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/accumulation_twitter_quality


# BASELINE
##############
#dirs_pattern="eval_results/old_results/human_ai_ratio_*reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/*"
#python visualize.py  --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_reddit_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_reddit_quality
#
#dirs_pattern="eval_results/old_results/human_ai_ratio_dataset_senator_tweets_split_all_acc_1_ft_size_4000_mixed_participants_1/*"
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_twitter_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_twitter_quality
#
#dirs_pattern=" eval_results/old_results/human_ai_ratio_dataset_100m_tweets_*participants_1/*"
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_100m_twitter_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_100m_twitter_quality

# interaction plots
dirs_pattern=""
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_dataset_webis_reddit_split_test_acc_1_ft_size_4000_mixed_participants_1/*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_dataset_senator_tweets_split_all_acc_1_ft_size_4000_mixed_participants_1/*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_dataset_100m_tweets_*participants_1/*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_dataset_reddit_submissions_*participants_1/*"
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_100m_twitter_diverisity
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_100m_twitter_diverisity
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_diverisity
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_250 --part part_0 -ip -ig 19 -igs 5 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/baseline_quality
exit


# Toxicity
#############
#dirs_pattern="eval_results/old_results/human_ai_ratio_*reddit*pants_1/*"
#python visualize.py --directories $dirs_pattern --metric toxicity --part part_0 --assert-n-datapoints 5 -ip -ig 19 -igs 5 -fhr 1.0 --no-show --save-path ${plots_dir}/toxicity_reddit_ip
#python visualize.py --directories $dirs_pattern --metric cos_diversity_stella_cap_250 --part part_0 --assert-n-datapoints 5 -ip -ig 19 -igs 5 -fhr 1.0 --no-show --save-path ${plots_dir}/toxicity_ref_div_reddit_ip
#python visualize.py --directories $dirs_pattern --metric llama_quality_cap_100 --part part_0 --assert-n-datapoints 5 -ip -ig 19 -igs 5 -fhr 1.0 --no-show --save-path ${plots_dir}/toxicity_ref_qual_reddit_ip
#python correlate_metrics.py  --directories $dirs_pattern --metrics toxicity llama_quality_cap_100 --no-show --save-path ${plots_dir}/correlations_tox_qual
#python correlate_metrics.py  --directories $dirs_pattern --metrics toxicity cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/correlations_tox_div
#exit

#
#dirs_pattern="eval_results/old_results/human_ai_ratio_*reddit*acc_0*pants_1/*"
#python visualize.py --directories $dirs_pattern --metric toxicity --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/toxicity_reddit_replace
#
#dirs_pattern="eval_results/old_results/human_ai_ratio_*reddit_split*acc_1*pants_1/*"
#python visualize.py --directories $dirs_pattern --metric toxicity --part part_0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/toxicity_reddit_baseline
#
#dirs_pattern="eval_results/old_results/human_ai_ratio_*tweet*pants_1/*"
#python visualize.py --directories $dirs_pattern --metric toxicity --part part_0 --assert-n-datapoints 5 -ip -ig 19 -igs 5 -fhr 1.0 --no-show --save-path ${plots_dir}/toxicity_twitter_ip


# correlate generations
#dirs_pattern=""
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*reddit*/generated_*"
#dirs_pattern+=" eval_results/old_results/human_ai_ratio_*tweet*/generated_*"
#python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_cap_100 --generations 9 19
#python correlate_metrics.py  --directories $dirs_pattern --metrics cos_diversity_stella cos_diversity_stella_cap_250 --no-show --save-path ${plots_dir}/correlations_div
#exit


#python correlate_metrics.py  --directories $dirs_pattern --metrics llama_quality_cap_4000 llama_quality_cap_100 --no-show --save-path ${plots_dir}/correlations_llama_llama
#python correlate_metrics.py  --directories $dirs_pattern --metrics gpt4o-mini_quality_cap_100 llama_quality_cap_100 --no-show --save-path ${plots_dir}/correlations_gpt_llama


## Define metrics
#metrics=(
##   quality
#  "gpt4o-mini_quality_cap_100"
#  "llama_quality_cap_100"
#  "llama_quality_cap_4000"
#  "aggregate_reading_level"
#  "n_unique_words_per_post"
##   diversity
#  "total_unique_words_cap_4000"
#  "cos_diversity_stella"
#  "n_unique_posts"
#  "cos_diversity_minilm"
#)
#
#for metric in "${metrics[@]}"; do
#  echo $metric
#  python visualize.py --directories $dirs_pattern --metric $metric --part part_0 -ip -ig 19 -fhr 1.0 --assert-n-datapoints 5 --no-show --save-path ${plots_dir}/${metric}_ip
#done
#
## ft setup
#######################
##dirs_pattern+=""
##dirs_pattern+=" eval_results/ft_setup_results/SETUP_FT_SAME_INSTR*reddit*/generated_*"
##dirs_pattern+=" eval_results/ft_setup_results/SETUP_FT_SAME_INSTR*tweets*/generated_*"
##python visualize_ft.py  --directories $dirs_pattern --metric llama_quality_cap_8000 --assert-n-datapoints 5
##exit
#