
# llama
# dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_split_test_*ft_size_*_Me*_participants_*_roof_prob_0.03/generated_*"

# compare models
# dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_split_test_*ft_size_4000_*_participants_[12]_roof_prob_0.03/generated_*"
# dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_split_test_*ft_size_4000_Met*_participants_2_roof_prob_0.03/generated_*"


dirs_pattern="eval_results/dev_results/human_ai_ratio_v3_epochs_0.001_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_*"
dirs_pattern="eval_results/dev_results/human_ai_ratio_v3_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_[24]*"

dirs_pattern="eval_results/dev_results/human_ai_ratio_v3_acc_1_epochs_1_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_*"

dirs_pattern="eval_results/dev_results/human_ai_ratio_v3_*_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_*"

#plots_dir="./viz_results/compare_models"
plots_dir="./viz_results/accumulate_replace_early_stop"
mkdir -p $plots_dir

dirs_pattern=""
# accumulate
dirs_pattern+=" eval_results/dev_results/human_ai_ratio_v3_acc_1_epochs_*participants_1_roof_prob_0.03/generated_*"
# replace
#dirs_pattern+=" eval_results/dev_results/human_ai_ratio_v3_split*participants_2_roof_prob_0.03/generated_*"
## no train
#dirs_pattern+=" eval_results/dev_results/human_ai_ratio_v3_epochs_0*participants_2_roof_prob_0.03/generated_*"

python visualize.py  --directories $dirs_pattern --metric normalized_levenshtein_diversity --part part_0
#python visualize.py  --directories $dirs_pattern --metric input_ai_ratio --part part_0
exit

# Define metrics
metrics=(
  "uniq"
  "div"
  "tox"
  "ttr"
  "mttr"
  "n_words"
  "ce"
)

for metric in "${metrics[@]}"; do
  echo $metric
  python visualize.py --directories $dirs_pattern --metric $metric --part part_0 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/${metric}
  python visualize.py --directories $dirs_pattern --metric $metric --part part_0 --per-seed --no-show --save-path ${plots_dir}/${metric}_per_seed
  python visualize.py --directories $dirs_pattern --metric $metric --part part_0 -ip -ipg 19 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/${metric}_ip
done
