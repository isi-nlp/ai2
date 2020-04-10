"""
Distance measurer class - need to initialize with the desired distance name
"""

import torch

distance_supported = {'cosine', 'manhattan', 'euclidean'}


class DistanceMeasurer:

    def __init__(self, distance_name):
        assert distance_name in distance_supported, 'Distance selected not supported by the helper class yet.'
        self.distance_name = distance_name

    def get_distance(self, embedding_1, embedding_2):

        # Make sure embedding_1 and embedding_2 are torch tensors
        if not isinstance(embedding_1, torch.Tensor):
            embedding_1 = torch.tensor(embedding_1, dtype=torch.float64)
        if not isinstance(embedding_2, torch.Tensor):
            embedding_2 = torch.tensor(embedding_2, dtype=torch.float64)

        assert embedding_1.size() == embedding_2.size(), 'Embedding size does not equal to each other'

        if self.distance_name == 'cosine':
            return self.cosine_dist(embedding_1, embedding_2)
        elif self.distance_name == 'manhattan':
            return self.manhattan_dist(embedding_1, embedding_2)
        elif self.distance_name == 'euclidean':
            return self.euclidean_dist(embedding_1, embedding_2)
        else:
            raise NotImplementedError("You really shouldn't have reached this line.")

    @staticmethod
    def cosine_dist(embed_1, embed_2):
        return torch.dot(embed_1, embed_2)/max(torch.norm(embed_1, 2)*torch.norm(embed_2, 2), 1e-08)

    @staticmethod
    def manhattan_dist(embed_1, embed_2):
        return torch.norm(embed_1 - embed_2, 1)

    @staticmethod
    def euclidean_dist(embed_1, embed_2):
        return torch.norm(embed_1 - embed_2, 2)

