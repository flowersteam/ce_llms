from pathlib import Path
import json
import argparse
from collections import defaultdict
import pickle

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
print("N generations: ", n_generations)


# Store datasets to avoid recomputing
datasets_savepath = f'{eval_save_dir}/datasets.pkl'

try:
    with open(datasets_savepath, 'rb') as f:
        datasets = pickle.load(f)
    print(f"Loaded datasets from pickle")
except:
    print(f"Adding bert embeddings")
    for gen_i in range(0, n_generations):
        print(f"Gen {gen_i}/{n_generations-1}")
        gen_csv = experiment_dir / f"gen_{gen_i}" / f"{participant}/generations.csv"
        ai_dataset = load_dataset_from_csv(gen_csv)
        ai_dataset = bert_embedder.add_bert_embeddings(ai_dataset)
        datasets.append(ai_dataset)
        #store to pickle
        with open(datasets_savepath, 'wb') as f:
            pickle.dump(datasets, f)
    print(f"Saved datasets to pickle: {datasets_savepath}")


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


for i, d in enumerate(datasets):
    print(f'datasets {i}/{len(datasets)}')
    embs = np.array(d['embeddings'])

    ## Storing full data to avoid recomputing each time

    
    try: 
        with open(f'{eval_save_dir}/var_diversities_gen{i}.pickle', 'rb') as f:
            var_diversities = pickle.load(f)
        print(f"Loaded var_diversities from pickle")
    except:
        var_diversities = compute_var_diveristy(embs)
        #store to pickle
        with open(f'{eval_save_dir}/var_diversities_gen{i}.pickle', 'wb') as f:
            pickle.dump(var_diversities, f)
        print(f"Saved var_diversities to pickle")

    results['var_diversities'].append(var_diversities)


    try:
        with open(f'{eval_save_dir}/cos_diversities_gen{i}.pickle', 'rb') as f:
            cos_diversities = pickle.load(f)
        print(f"Loaded cos_diversities from pickle")
    except:
        cos_diversities = compute_cos_diveristy(embs)
        #store to pickle
        with open(f'{eval_save_dir}/cos_diversities_gen{i}.pickle', 'wb') as f:
            pickle.dump(cos_diversities, f)
        print(f"Saved cos_diversities to pickle")

    results['cos_diversities'].append(cos_diversities)



    if args.human_dataset:
        loss, acc = fit_logreg(
            np.array(human_dataset['embeddings']),
            embs, max_iter=100
        )
        results['logreg_loss'].append(loss)
        results['logreg_accuracy'].append(acc)




    try:
        with open(f'{eval_save_dir}/ttrs_gen{i}.pickle', 'rb') as f:
            ttrs = pickle.load(f)
        print(f"Loaded ttrs from pickle")
    except:
        ttrs = [calculate_ttr(tx) for tx in d['text']]
        #store to pickle
        with open(f'{eval_save_dir}/ttrs_gen{i}.pickle', 'wb') as f:
            pickle.dump(ttrs, f)
        print(f"Saved ttrs to pickle")
    try:
        with open(f'{eval_save_dir}/n_words_gen{i}.pickle', 'rb') as f:
            n_words = pickle.load(f)
        print(f"Loaded n_words from pickle")
    except:
        n_words = [num_words(tx) for tx in d['text']]
        #store to pickle
        with open(f'{eval_save_dir}/n_words_gen{i}.pickle', 'wb') as f:
            pickle.dump(n_words, f)
        print(f"Saved n_words to pickle")
    try:
        with open(f'{eval_save_dir}/positivity_gen{i}.pickle', 'rb') as f:
            positivity = pickle.load(f)
        print(f"Loaded positivity from pickle")
    except:
        positivity = [get_positivity(tx) for tx in d['text']]
        #store to pickle
        with open(f'{eval_save_dir}/positivity_gen{i}.pickle', 'wb') as f:
            pickle.dump(positivity, f)
        print(f"Saved positivity to pickle")

    results['mean_ttrs'].append(ttrs)
    results['mean_n_words'].append(n_words)
    results['positivity'].append(positivity)


    try:
        with open(f'{eval_save_dir}/dataset_lens_gen{i}.pickle', 'rb') as f:
            dataset_lens = pickle.load(f)
        print(f"Loaded dataset_lens from pickle")
    except:
        dataset_lens = len(d['text'])
        #store to pickle
        with open(f'{eval_save_dir}/dataset_lens_gen{i}.pickle', 'wb') as f:
            pickle.dump(dataset_lens, f)
        print(f"Saved dataset_lens to pickle")
    results['dataset_lens'].append(dataset_lens)



    try: 
        with open(f'{eval_save_dir}/toxicity_gen{i}.pickle', 'rb') as f:
            toxicity = pickle.load(f)
        print(f"Loaded toxicity from pickle")
    except:
        print("computing toxicity...")
        toxicity = get_toxicity_batch(d['text'])
        #store to pickle
        with open(f'{eval_save_dir}/toxicity_gen{i}.pickle', 'wb') as f:
            pickle.dump(toxicity, f)
        print(f"Saved toxicity to pickle")

    results['toxicity'].append(toxicity)

    try:
        with open(f'{eval_save_dir}/political_bias_gen{i}.pickle', 'rb') as f:
            political_bias = pickle.load(f)
        print(f"Loaded political_bias from pickle")
    except:
        print("computing political bias...")
        political_bias = get_political_bias_batch(d['text'])
        #store to pickle
        with open(f'{eval_save_dir}/political_bias_gen{i}.pickle', 'wb') as f:
            pickle.dump(political_bias, f)
        print(f"Saved political_bias to pickle")


    results['political_bias'].append(political_bias)


    if args.ppl:
        results['ppls'].append(ppl_metric.evaluate(d['text'], bs=500))


results_path = eval_save_dir / 'results.json'
os.makedirs(results_path.parent, exist_ok=True)
with open(results_path, 'w') as results_file:
    json.dump(results, results_file, indent=6)

print(f'Metrics saved to: {results_path}')


if args.visualize_datasets:
    visualize_datasets(datasets, dataset_labels, experiment_tag)
