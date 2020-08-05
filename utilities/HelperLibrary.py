"""
Library of Helper Functions
transform: takes a task formula and turn into a lambda function to processes task stories
distance functions: Measure the distance of two embedding of the same size
"""
from itertools import cycle
from typing import Union

import torch


# Create a lambda function that parse a row in the data frame representation of an AI2 task
def transform(formula):
    parsed_context, parsed_choices = formula.split("->")
    # alphanli:     (obs1 + obs2 -> hyp1|hyp2)
    # c_hellaswag:  (ctx_a + ctx_b -> opt0|opt1|opt2|opt3)
    # physicaliqa:  (goal -> sol1|sol2)
    # socialiqa:    (context + question -> answerA|answerB|answerC)
    parsed_context = parsed_context.strip().split("+")
    parsed_choices = parsed_choices.strip().split("|")

    # Apply to a pandas data frame row
    def wrapper(row):
        context = " ".join(row[a_context.strip()] for a_context in parsed_context)
        choices = [row[a_choice.strip()] for a_choice in parsed_choices]
        return list(zip(cycle([context]), choices))

    return wrapper


# Cosine Distance
def cosine_dist(embed_1: torch.Tensor, embed_2: torch.Tensor):
    dist = torch.tensor(1) - torch.dot(embed_1, embed_2) / max(torch.norm(embed_1, 2) * torch.norm(embed_2, 2), 1e-08)
    return dist.item()


# All l_p norm distance. l=1 is manhattan, l=2 is euclidean
def l_norm_dist(embed_1: torch.Tensor, embed_2: torch.Tensor, l_norm: Union[int, float] = 2):
    dist = torch.norm(embed_1 - embed_2, p=l_norm)
    return dist.item

