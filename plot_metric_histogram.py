import matplotlib.pyplot as plt
import numpy as np
from datasets import concatenate_datasets, load_from_disk, DatasetDict

metric = 'llama_quality_scale'

names = []
ds = []
# senators
names.append(f"senator_tweets_{metric}")
ds.append(load_from_disk("data/senator_tweets/prepared-senator-tweets-qualities/"))

# submissions
names.append(f"reddit_submissions_{metric}")
ds.append(load_from_disk("./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"))

names.append(f"100M_tweets_{metric}")
ds.append(load_from_disk("data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus/"))

names.append(f"webis_{metric}")
ds.append(load_from_disk("data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"))

for name, d in zip(names, ds):
    if type(d) == DatasetDict:
        assert d.keys() == {'train', 'test'}
        d = concatenate_datasets([d['train'], d['test']])

    quality_scores = d[metric]
    quality_scores = [s if s else -10 for s in quality_scores]  # rejected

    # Calculate terciles
    tercile1 = np.percentile(quality_scores, 33.33)
    tercile2 = np.percentile(quality_scores, 66.67)
    half = np.percentile(quality_scores, 50.00)

    plt.figure(figsize=(12, 6))
    plt.hist(quality_scores, bins=110, range=(-10, 100), color='blue', alpha=0.7)

    # Add tercile lines
    plt.axvline(x=tercile1, color='r', linestyle='--', label=f'33rd percentile: {tercile1:.1f}')
    plt.axvline(x=half, color='b', linestyle='-', label=f'50th percentile: {half:.1f}')
    plt.axvline(x=tercile2, color='g', linestyle='--', label=f'67th percentile: {tercile2:.1f}')

    # Add labels and styling
    plt.title(name, fontsize=14, pad=20)
    plt.xlabel('Quality Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(f'viz_results/histograms/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()