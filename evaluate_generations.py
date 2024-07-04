from pathlib import Path
import json
import argparse
from collections import defaultdict

from dataset_utils import *
from eval_utils import *
from visualization_utils import *
# from evaluate import load


hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"

parser = argparse.ArgumentParser()
parser.add_argument("--experiment-dir", type=str, required=True)
parser.add_argument("--ppl", action="store_true", help="Compute perplexity")
parser.add_argument("--visualize-datasets", action="store_true", help="Visualize the datasets")
parser.add_argument("--human-dataset", action="store_true", help="Analyze human dataset")
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


# load bert model
print("Loading bert")
bert_embedder = BertEmbedder()


gen_0_part_0_dataset = json.loads((experiment_dir / "gen_0" / "part_0" / "log.json").read_text(encoding="UTF-8"))['args']['dataset']
assert gen_0_part_0_dataset == "twitter"

if args.human_dataset:
    # Load and encode human dataset
    print("Load and encode human dataset")
    # human dataset
    print("Loading human dataset")
    human_dataset, _, feat_sentiment = load_twitter_dataset(cache_dir=hf_cache_dir, load_n=4000)
    print("Encoding human dataset")
    human_dataset = bert_embedder.add_bert_embeddings(human_dataset)
    #
    human_dataset = human_dataset.sort('Political Lean')
    human_dataset_labels = ["Human"]
    datasets = [human_dataset]

    # human_dataset_dem = human_dataset.filter(lambda ex: ex['Political Lean'] == 0)
    # human_dataset_rep = human_dataset.filter(lambda ex: ex['Political Lean'] == 1)
    # human_dataset_labels = ["Human Dem", "Human Rep"]
    # datasets = [human_dataset_dem, human_dataset_rep]
else:
    print("Skipping human dataset")
    # no human dataset human_dataset = None
    human_dataset_labels = []
    datasets = []


print("Load and encode AI datasets")

participant = "part_0"
n_samples = 5
n_generations = len(list(experiment_dir.glob("gen_[0-9]*")))

print(f"Adding bert embeddings")
for gen_i in range(0, n_generations):
    print(f"Gen {gen_i}/{n_generations-1}")
    gen_csv = experiment_dir / f"gen_{gen_i}" / f"{participant}/generations.csv"
    ai_dataset = load_dataset_from_csv(gen_csv)
    ai_dataset = bert_embedder.add_bert_embeddings(ai_dataset)
    datasets.append(ai_dataset)

dataset_labels = human_dataset_labels + [f"AI gen {i}" for i in range(len(datasets)-len(human_dataset_labels))]

# Show random samples
for lab, d in zip(dataset_labels, datasets):
    print(f"{lab} random samples")
    samples = d.shuffle().select(range(n_samples))
    for sample in samples:
        print("\tSample:", sample['text'])


# Evaluate Metrics

results = defaultdict(list)
results["dataset_labels"] = dataset_labels

if args.ppl:
    ppl_model = 'mistralai/Mistral-7B-v0.1'
    ppl_metric = Perplexity(ppl_model)


for d in datasets:
    embs = np.array(d['embeddings'])
    results['var_diversities'].append(compute_var_diveristy(embs))
    results['cos_diversities'].append(compute_cos_diveristy(embs))

    if args.human_dataset:
        loss, acc = fit_logreg(
            np.array(human_dataset['embeddings']),
            embs, max_iter=100
        )
        results['logreg_loss'].append(loss)
        results['logreg_accuracy'].append(acc)

    results['mean_ttrs'].append(np.mean([calculate_ttr(tx) for tx in d['text']]))
    results['mean_n_words'].append(np.mean([num_words(tx) for tx in d['text']]))
    results['positivity'].append(np.mean([get_positivity(tx) for tx in d['text']])) 
    results['dataset_lens'].append(len(d['text']))



    if args.ppl:
        results['ppls'].append(ppl_metric.evaluate(d['text'], bs=500))


results_path = eval_save_dir / 'results.json'
os.makedirs(results_path.parent, exist_ok=True)
with open(results_path, 'w') as results_file:
    json.dump(results, results_file, indent=6)

print(f'Metrics saved to: {results_path}')


if args.visualize_datasets:
    visualize_datasets(datasets, dataset_labels, experiment_tag)
