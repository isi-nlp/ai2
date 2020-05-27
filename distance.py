"""
After parsing embeddings, the script compare train embedding to the dev embedding and write out top correlation
entries
"""

import numpy as np
import torch
import tqdm

from utilities.DistanceMeasurer import DistanceMeasurer

# Control Variables
TOP_N = 5
TOTAL_SEED = 10
correct_range = range(16965, 33931)
dm = DistanceMeasurer('cosine')
EMBED_DIR = r"/nas/home/dwangli/colleagues/mzhang/ai2/embeddings/final/"

# Init accumulator array
over_all_accuracies = torch.zeros(TOTAL_SEED, dtype=torch.float)

for a_seed in range(TOTAL_SEED):

    print(f"\nSeed {a_seed}")

    # Load in the embedding arrays
    train_embed = np.load(EMBED_DIR + f"roberta-large-alphanli_embed_20_ooa_train-s{a_seed}.npy")
    dev_embed = np.load(EMBED_DIR + f"roberta-large-alphanli_embed_20_ooa_dev-s{a_seed}.npy")

    # Loop through the train embeddings to get distances for each embedding
    train_distances = []
    for idx_train, a_train_embed in tqdm.tqdm(enumerate(train_embed), total=len(train_embed)):
        distances = torch.tensor([dm.get_distance(a_train_embed, a_dev_embed) for a_dev_embed in dev_embed])
        train_distances.append((idx_train, torch.mean(distances)))

    # Sort the result of the train distance from the closest to the largest
    train_distances.sort(key=lambda tup: tup[1])

    # Print out results and get the number of correct ones for this seed
    num_correct = 0
    print("On Average Closest Train Stories are:")
    print('StoryID\tCosineDist')
    for idx, cosine_distance in train_distances[:TOP_N]:
        print(f'{idx}\t{cosine_distance}')
        if idx in correct_range:
            num_correct += 1

    over_all_accuracies[a_seed] = num_correct/TOP_N

print("Mean Accuracy with 2 Standard Deviation")
print(f"{over_all_accuracies.mean():.3f} +/- {2 * over_all_accuracies.std():.3f}")
