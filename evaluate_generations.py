from pathlib import Path
import argparse
from collections import defaultdict

import datasets


from dataset_utils import *
from eval_utils import *
from visualization_utils import *

start_time = time.time()

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = os.environ["HF_HOME"]

parser = argparse.ArgumentParser()
parser.add_argument("--seed-dir", type=str, help="Experiment directory")
parser.add_argument("--emb", action="store_true", help="Embeddings and embedding based metrics")
parser.add_argument("--gpt-quality", action="store_true", help="GPT quality")
parser.add_argument("--llama-quality", action="store_true", help="Judge quality")
parser.add_argument("--input", action="store_true", help="Judge quality")
parser.add_argument("--gib", action="store_true", help="Compute gibberish score")
parser.add_argument("--pol", action="store_true", help="Compute political lean")
parser.add_argument("--tox", action="store_true", help="Compute toxicity")
parser.add_argument("--pos", action="store_true", help="Compute positivity")
parser.add_argument("--cap", type=int, default=250)
parser.add_argument("--n-samples-to-show", type=int, default=0, help="Show n random samples")
parser.add_argument("--visualize-datasets", action="store_true", help="Visualize the datasets")
parser.add_argument("--participant", type=str, default="all_parts")
parser.add_argument('--generations', nargs='+', type=int, help='Generations to evaluate', default=None)
args = parser.parse_args()
print(args)

print(f"Participant: {args.participant}")
seed_dir = Path(args.seed_dir)

print(f"Experiment dir: {seed_dir}")

n_samples_cap = args.cap

tokens_total = 0
spent_total = 0

eval_save_dir = Path("./eval_results") / seed_dir / f"{args.participant}"
eval_save_dir.mkdir(parents=True, exist_ok=True)

cache_dir = Path(".cache") / eval_save_dir
cache_dir.mkdir(parents=True, exist_ok=True)

stella_embedder = None
modernbert_embedder = None

if args.participant == "all_parts":
    n_parts = len(list(seed_dir.glob(f"gen_0/part_[0-9]*/full_output_dataset/data-00000-of-00001.arrow")))
    all_participants = [f"part_{p_i}" for p_i in range(n_parts)]
else:
    all_participants = [args.participant]

print("Participants:", all_participants)

if args.generations is None:
    n_generations = len(list(seed_dir.glob(f"gen_[0-9]*/{all_participants[-1]}/full_output_dataset/data-00000-of-00001.arrow")))
    assert n_generations == 20

    generations = list(range(n_generations))
else:
    generations = list(args.generations)

print("Evaluating generations: ", generations)


results = defaultdict(dict)

if args.input:
    print(f"Input dataset")
    # load input dataset
    ####################

    input_ds = []
    for part in all_participants:
        part_input_d_path = str(seed_dir / f"gen_0" / f"{part}/input_dataset")
        part_input_d = datasets.load_from_disk(part_input_d_path)

        if args.emb:
            emb_column_name = f"stella_embeddings"
            if emb_column_name not in part_input_d.features:
                stella_embedder = StellaEmbedder() if stella_embedder is None else stella_embedder
                part_input_d = stella_embedder.add_embeddings(part_input_d, batch_size=256)
                overwrite_to_disk(part_input_d, part_input_d_path)

        input_ds.append(part_input_d)

    input_d = concatenate_datasets(input_ds)

    # compute metrics
    #################
    capped_input_d = input_d.select(range(n_samples_cap))

    quick_metrics_results = get_or_compute_cache(
        cache_path=str(cache_dir / f'input_quick_metrics_cap_{n_samples_cap}_gen_0_part_{args.participant}.pickle'),
        compute_fn=compute_quick_metrics, input_d=capped_input_d
    )

    for quick_metric, quick_metric_scores in quick_metrics_results.items():
        results[f"input_{quick_metric}_cap_{n_samples_cap}"][0] = quick_metric_scores

    if args.emb:
        emb_name="stella"
        embs = np.array(capped_input_d[f"{emb_name}_embeddings"])

        results[f'input_cos_diversity_{emb_name}_cap_{n_samples_cap}'][0] = get_or_compute_cache(
            cache_path=str(cache_dir / f'{emb_name}_cos_diversities_cap_{n_samples_cap}_input_gen_0_part_{args.participant}.pickle'),
            compute_fn=compute_cos_diveristy, embs=embs
        )

    if args.llama_quality:
        results[f'input_llama_quality_cap_{n_samples_cap}'][0] = get_or_compute_cache(
            cache_path=str(cache_dir / f'llama_quality_cap_{n_samples_cap}_input_gen_0_part_{args.participant}.pickle'),
            compute_fn=llama_quality, texts=capped_input_d['text']
        )

    if args.gib:
        results[f'input_gibberish_score_cap_{n_samples_cap}'][0] = get_or_compute_cache(
            cache_path=str(cache_dir / f'gibberish_score_cap_{n_samples_cap}_gen_0_part_{args.participant}.pickle'),
            compute_fn=get_gibberish_scores, texts=capped_input_d['text']
        )


# output datasets
#################
# load datasets
all_datasets_capped = {}
print(f"Loading datasets and adding embeddings")
for gen_i in generations:
    print(f"Gen {gen_i}")
    all_datasets_capped[gen_i] = get_or_compute_cache(
        cache_path=str(cache_dir / f"datasets_emb_{args.emb}_cap_{n_samples_cap}_gen_{gen_i}_part_{args.participant}.pkl"),
        compute_fn=load_merged_participants_dataset,
        path_pattern=str(seed_dir / f"gen_{gen_i}" / f"{args.participant}/full_output_dataset"),
        all_participants=all_participants,
        size=n_samples_cap
    )

print(f"Adding embeddings")
all_datasets_capped_ = {}
for d_gen_i, output_dataset_capped in all_datasets_capped.items():
    print(f"Adding embeddings: Gen {d_gen_i}")

    if args.emb:
        if "stella_embeddings" not in output_dataset_capped.features:
            stella_embedder = StellaEmbedder() if stella_embedder is None else stella_embedder
            output_dataset_capped = stella_embedder.add_embeddings(output_dataset_capped, batch_size=256)

    all_datasets_capped_[d_gen_i] = output_dataset_capped

all_datasets_capped = all_datasets_capped_

assert len(all_datasets_capped) == len(generations)
for gen_i, d in all_datasets_capped.items():
    dataset_cache_path = cache_dir / f"datasets_emb_{args.emb}_cap_{n_samples_cap}_gen_{gen_i}_part_{args.participant}.pkl"
    with open(dataset_cache_path, 'wb') as f:
        pickle.dump(d, f)
    print(f"Saved datasets to pickle: {dataset_cache_path}")


# Show random samples
if args.n_samples_to_show > 0:
    for d_gen_i, d in all_datasets_capped.items():
        print(f"Dataset from gen: {d_gen_i} random samples")
        samples = d.shuffle().select(range(args.n_samples_to_show))
        for sample in samples:
            print("\tSample:", sample['text'])


# Evaluate datasets
###################

# overall metrics
sample_datasets_args = json.loads((seed_dir / "gen_0" / "log_sample_datasets.json").read_text(encoding="UTF-8"))['args']
results["sample_datasets"][0] = sample_datasets_args

human_dataset = sample_datasets_args['human_dataset']
n_participants = sample_datasets_args['n_participants']
results["human_dataset"] = human_dataset
results["n_participants"] = n_participants

# extract the number of human and ai posts in the training set of each generation
gen_n, human_n = get_ai_human_n_posts(seed_dir, gen_i=1)
results["gen_n"] = gen_n
results["human_n"] = human_n

ai_ratio = gen_n / (human_n + gen_n)
results["ai_ratio"] = ai_ratio
results["gen_train_ratio"] = sample_datasets_args.get('gen_train_dataset_size_ratio', 1.0)

# assert that all generations have the same ai_human ratio
for gen_i in generations:
    if gen_i == 0: continue
    assert get_ai_human_n_posts(seed_dir, gen_i=gen_i) == (gen_n, human_n)

# per generation metrics
for d_gen_i, d in all_datasets_capped.items():
    print(f'Output dataset {d_gen_i}')

    # quick metrics
    print("Computing quick metrics")
    quick_metrics_results = get_or_compute_cache(
        cache_path=str(cache_dir / f'quick_metrics_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
        compute_fn=compute_quick_metrics, input_d=d
    )

    for quick_metric, quick_metric_scores in quick_metrics_results.items():
        results[f"{quick_metric}_cap_{n_samples_cap}"][d_gen_i] = quick_metric_scores

    if args.emb:
        print("Computing diversity")
        emb_name="stella"
        emb_column_name = f"{emb_name}_embeddings"
        embs = np.array(d[emb_column_name])

        results[f'var_diversity_{emb_name}_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f"{emb_name}_var_diversities_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle"),
            compute_fn=compute_var_diveristy, embs=embs
        )

        results[f'cos_diversity_{emb_name}_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f"{emb_name}_cos_diversities_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle"),
            compute_fn=compute_cos_diveristy, embs=embs
        )


    if args.llama_quality:
        results[f'llama_quality_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f'llama_quality_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=llama_quality, texts=d['text']
        )

    if args.gpt_quality:
        gpt_judge_cap = 100

        gpt_qualities, tokens = get_or_compute_cache(
            cache_path=str(cache_dir / f'gpt_4o_quality_cap_{gpt_judge_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=gpt4o_quality, texts=d.select(range(gpt_judge_cap))['text']
        )
        spent = (tokens/1_000_000)*0.15
        tokens_total += tokens
        spent_total += spent

        results[f'gpt4o-mini_quality_cap_{gpt_judge_cap}'][d_gen_i] = gpt_qualities
        results[f'gpt4o-mini_tokens_cap_{gpt_judge_cap}'][d_gen_i] = tokens
        results[f'gpt4o-mini_tokens_total_cap_{gpt_judge_cap}'][d_gen_i] = tokens_total
        results[f'gpt4o-mini_spent_cap_{gpt_judge_cap}'][d_gen_i] = spent
        results[f'gpt4o-mini_spent_total_cap_{gpt_judge_cap}'][d_gen_i] = spent_total

    if args.gib:
        results[f'gibberish_score_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f'gibberish_score_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=get_gibberish_scores, texts=d['text']
        )

    if args.pos:
        results[f'positivity_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f'positivity_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=get_positivites, texts=d['text']
        )

    if args.tox:
        results[f'toxicity_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f'toxicity_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=get_toxicity_batch, texts=d['text'], batch_size=1024
        )

    if args.pol:
        results[f'llama_pol_cap_{n_samples_cap}'][d_gen_i] = get_or_compute_cache(
            cache_path=str(cache_dir / f'llama_pol_cap_{n_samples_cap}_gen_{d_gen_i}_part_{args.participant}.pickle'),
            compute_fn=llama_pol_lean, texts=d['text']
        )


results_path = eval_save_dir / 'results.json'
os.makedirs(results_path.parent, exist_ok=True)

if results_path.is_file():
    # add new results to the old results, and replace the new results with the merged result ones
    with open(results_path, "r") as results_file:
        previous_results = json.load(results_file)

    for metric in results:
        # if metric not in previous_results or metric == "llama_quality_cap_4000":
        if metric not in previous_results:
            # add new metric to results
            previous_results[metric] = results[metric]
        else:
            # merge metrics
            previous_scores = previous_results[metric]
            new_scores = results[metric]

            if isinstance(new_scores, dict):

                # backwards compatibility
                if isinstance(previous_scores, list):
                    previous_results[metric] = dict(enumerate(previous_results[metric]))

                # keys should already be ints but just to make sure
                previous_results[metric] = {int(k): v for k, v in previous_results[metric].items()}
                new_scores = {int(k): v for k, v in new_scores.items()}

                for g, s in new_scores.items():
                    previous_results[metric][g] = s

            else:
                # e.g. ratio
                previous_results[metric] = new_scores

    results = previous_results

with open(results_path, 'w') as results_file:
    json.dump(results, results_file, indent=6)

print(f'Metrics saved to: {results_path}')

if args.visualize_datasets:
    dataset_labels = [f"AI gen {i}" for i in range(len(all_datasets_capped.items()))]
    visualize_datasets(all_datasets_capped, dataset_labels, seed_dir)

end_time = time.time()
total_time = end_time - start_time
print(f"Total time (evaluate generations): {total_time}s")