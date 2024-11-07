import datasets
# import itertools
# import numpy as np
# from collections import Counter
#
# from itertools import combinations
import glob
from eval_utils import Perplexity

# webis_reddit_dataset_path = "./data/webis/prepared-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17"
# d = datasets.load_from_disk(webis_reddit_dataset_path)

d_path=glob.glob("dev_results/human_ai_ratio_no_tldr_v2_rank_16_alpha_16_rslora_False_bs_16_lr_2e-4_lr_sched_linear_warmup_ratio_0.00125_temp_1.5_min_p_0.2_webis_reddit_ft_size_4000_Meta-Llama-3.1-8B_participants_2_roof_prob_0.03/generated_1000_human_3000_unsloth/seed_2_2024-11-04_14-33-34.062881747_2024-11-04_14-33-34/gen_19/part_0/output_dataset")[0]
d = datasets.load_from_disk(d_path)

from IPython import embed; embed();

from eval_utils import get_toxicity_batch
toxicity = get_toxicity_batch(d['text'], batch_size=1024)

# perplexity
response_template = "### RESPONSE\n"
texts = [f"### INSTRUCTION\n{ins}\n{response_template}\n{tx}" for ins, tx in zip(d['instruction'], d['text'])]

perplexity = Perplexity('Qwen/Qwen2.5-72B', model_args={"device_map": "auto", "torch_dtype": "auto"})
# bigger model
ppl = perplexity.evaluate(
    texts, response_template=response_template,
    batch_size=4, add_start_token=False, max_length=1024, add_end_token=True
)["perplexities"]

from IPython import embed; embed();
