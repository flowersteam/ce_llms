plots_dir="./viz_results/test_split"
mkdir -p $plots_dir

#dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_*000_Meta-Llama-3.1-8B_participants_*_roof_prob_0.03/generated_*"
#dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_split_test_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_8000_Meta-Llama-3.1-8B_participants_1_roof_prob_0.03/generated_*"
dirs_pattern="eval_results/dev_results/human_ai_ratio_no_tldr_v2_split_test_*participants_*_roof_prob_0.03/generated_*"
# dataset lens
python visualize.py  --directories $dirs_pattern --metric uniq  --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/dataset_lens

# diversity
python visualize.py  --directories $dirs_pattern  --metric div --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/diversity
python visualize.py  --directories $dirs_pattern  --metric div --per-seed --no-show --save-path ${plots_dir}/diversity_ip_per_seed
python visualize.py  --directories $dirs_pattern  --metric div -ip -ipg 19 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/diversity_ip

# ce
python visualize.py  --directories $dirs_pattern  --metric ce  --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/ce
python visualize.py  --directories $dirs_pattern  --metric ce  --per-seed --no-show --save-path ${plots_dir}/ce_per_seed
python visualize.py  --directories $dirs_pattern  --metric ce  -ip -ipg 19 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/ce_ip

# tox
python visualize.py  --directories $dirs_pattern  --metric tox --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/tox
python visualize.py  --directories $dirs_pattern  --metric tox --per-seed --no-show --save-path ${plots_dir}/tox_per_seed
python visualize.py  --directories $dirs_pattern  --metric tox -ip -ipg 19 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/tox_ip

# ttr
python visualize.py  --directories $dirs_pattern  --metric ttr --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/ttr
python visualize.py  --directories $dirs_pattern  --metric ttr --per-seed --no-show --save-path ${plots_dir}/ttr_ip_per_seed
python visualize.py  --directories $dirs_pattern  --metric ttr -ip -ipg 19 --assert-n-datapoints 3 --no-show --save-path ${plots_dir}/ttr_ip
