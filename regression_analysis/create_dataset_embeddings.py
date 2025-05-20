import argparse
import numpy as np
import torch
import datasets
from sentence_transformers import SentenceTransformer


def compute_embeddings(dataset, save_embeddings_path=None):

    texts = dataset['text']

    embed_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    embed_model.max_seq_length = 512

    print("Embedding the full dataset")
    pool = embed_model.start_multi_process_pool()
    embeddings = embed_model.encode_multi_process(
        texts,
        pool=pool, batch_size=64,
        show_progress_bar=True, normalize_embeddings=True,
    )
    embed_model.stop_multi_process_pool(pool)

    if save_embeddings_path:
        with open(save_embeddings_path, "wb") as f:
            np.save(f, embeddings)
        print("Embeddings saved to: ", save_embeddings_path)

    return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clustering parameters")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--embeddings_path', type=str, required=True)
    args = parser.parse_args()
    print(args)

    print("Loading dataset from: ", args.dataset_path)
    dataset = datasets.load_from_disk(args.dataset_path)

    print("Adding embeddings")
    embeddings = compute_embeddings(dataset=dataset, save_embeddings_path=args.embeddings_path)
