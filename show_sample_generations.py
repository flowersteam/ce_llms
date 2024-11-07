import os
from pathlib import Path
import argparse
from dataset_utils import *

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = os.environ["HF_HOME"]

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-dir", type=str, required=True)
args = parser.parse_args()

# load and encode AI datasets
# experiment_dir = Path("./results/Testing_iterative_learning")
# experiment_dir = Path("./results/Testing_iterative_learning_n_4k")
# experiment_dir = Path("./results/Testing_iterative_learning_n_4k_temp_1.5")
# experiment_dir = Path("./results/Testing_iterative_learning_n_4000_temp_1.0_lean_Liberal")
# experiment_dir = Path("./results/Testing_iterative_learning_n_4000_temp_1.0_lean_Conservative")
# experiment_dir = Path("./results/Testing_iterative_learning_deduplicate_n_4000_temp_0.7")

experiment_dir = Path(args.experiment_dir)
experiment_tag = experiment_dir.name
eval_save_dir = Path("./eval_results") / experiment_tag


# Load and encode human dataset
datasets = []

participant = "part_0"
n_samples = 10
n_generations = len(list(experiment_dir.glob("gen_[0-9]*")))

for gen_i in range(0, n_generations):
    gen_csv = experiment_dir / f"gen_{gen_i}" / f"{participant}/generations.csv"
    ai_dataset = load_dataset_from_texts_from_csv(gen_csv)
    datasets.append(ai_dataset)

dataset_labels = [f"AI gen {i}" for i in range(len(datasets))]

# Show random samples
for lab, d in zip(dataset_labels, datasets):
    print(f"{lab} random samples")
    samples = d.shuffle().select(range(n_samples))
    for sample in samples:
        print("\tSample:", sample['text'])