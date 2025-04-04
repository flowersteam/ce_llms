
import numpy as np
import json

from pathlib import Path
import requests

from eval_utils import llama_quality_scale, get_or_compute_cache

from scipy.stats import spearmanr, pearsonr

json_path = "data/OpenMEVA_ROC/mans_roc.json"
if Path(json_path).exists():
    with open(json_path, "r") as f:
        mans_roc = json.load(f)
    print("JSON file loaded")
else:
    mans_roc = requests.get("https://huggingface.co/datasets/Jiann/OpenMEVA/raw/main/data/mans_data/mans_roc.json").json()
    with open(json_path, "w") as f:
        json.dump(mans_roc, f)
    print("JSON file saved")

# extract the texts and scores
texts = []
scores = []
for i, story in mans_roc.items():
    for generator in ['fusion', 's2s', 'gpt_kg', 'gpt', 'plan_write']:
        texts.append(story['prompt'] + story['gen'][generator]['text'])
        scores.append(np.mean(story['gen'][generator]['score']))


# evaluate the dataset with llama
cache_dir = Path(".cache") / "openmeva"
cache_dir.mkdir(parents=True, exist_ok=True)

llama_scores = get_or_compute_cache(
    cache_path=str(cache_dir / f'llama_quality_scale_openmeva.pickle'),
    force_recompute=True,
    compute_fn=llama_quality_scale, texts=texts
)

# compute the correlations
pearson_corr, _ = pearsonr(scores, llama_scores)
print(f"Pearson: {pearson_corr}")

spearman_corr, _ = spearmanr(scores, llama_scores)
print(f"Spearman: {spearman_corr}")

# results
# Pearson: 0.515684325694146
# Spearman: 0.5222974492891408
