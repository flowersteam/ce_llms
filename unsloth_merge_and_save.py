from unsloth import FastLanguageModel

model_name = "dev_results/longer_human_ai_ratio_webis_reddit_v2_temp_1.0_ft_size_4000_Meta-Llama-3.1-8B-bnb-4bit_particiapnts_1_generated_4000_human_0_roof_prob_0.03_unsloth/seed_0_2024-10-22_22-49-43_2024-10-22_22-49-43/gen_19/part_0/model/final/"
model, tokenizer = FastLanguageModel.from_pretrained(model_name)
model.save_pretrained_merged(
    "dev_results/longer_human_ai_ratio_webis_reddit_v2_temp_1.0_ft_size_4000_Meta-Llama-3.1-8B-bnb-4bit_particiapnts_1_generated_4000_human_0_roof_prob_0.03_unsloth/seed_0_2024-10-22_22-49-43_2024-10-22_22-49-43/gen_19/part_0/model/final_merged/",
    tokenizer,
    save_method="merged_16bit"
)