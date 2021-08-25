# jac: Copied from myself. Source of code:
#      the module isi_power_analyses.common
#      in github.com/isi-vista/common-sense-power-analysis
"""
A collection of functions for performing comparison tests on pairs of models using leaderboard data.
"""
import logging
from typing import List, Sequence, Tuple

import scipy.stats
from scipy.stats import binom
from statsmodels.stats.proportion import proportions_ztest as statsmodels_ztest
from statsmodels.stats.contingency_tables import mcnemar as _statsmodels_mcnemar


logger = logging.getLogger(__name__)


def fishers_test(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Compute the significance (p-value) for a comparison between the given two models using Fisher's exact test.

    We do this by setting up a contingency table like this:

    | rows: correctness / columns: model | model 1 | model 2 |
    | ---------------------------------- | ------- | ------- |
    |                            correct |       x |       y |
    |                          incorrect |       z |       w |

    Note that this is *very different* from the contingency table we use for McNemar's test in other parts of the code.

    Note also that this specifically performs the *two-tailed* version of Fisher's exact test.

    This returns a pair (odds ratio, p-value).
    """
    return scipy.stats.fisher_exact(
        [
            [n_model1_correct, n_model2_correct],
            [test_set_size - n_model1_correct, test_set_size - n_model2_correct],
        ],
        # jac: https://statisticsbyjim.com/hypothesis-testing/use-one-tailed-tests/
        # makes a persuasive case that we really want a two-sided hypothesis test, not a one-sided one
        alternative="two-sided",
    )


def binomial_difference_of_proportions_test(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Compute the significance (p-value) for a comparison between the given two models using a binomial test
    for a difference of proportions.

    We do this by setting up a contingency table like this:

    | rows: correctness / columns: model | model 1 | model 2 |
    | ---------------------------------- | ------- | ------- |
    |                            correct |       x |       y |
    |                          incorrect |       z |       w |

    Note that this is *very different* from the contingency table we use for McNemar's test in other parts of the code.

    Note also that this specifically performs the *two-tailed* version of the difference of proportions test.

    This returns a pair (z-value, p-value).
    """
    # Handle edge case where models have identical accuracy
    # In this case the test statistic should be 0 (inasmuch as it make sense)
    # and the p-value should be 1
    # statsmodels can handle this when n_model_correct is positive and less than the test set size,
    # but not when it is 0 or equal to the test set size, so for simplicity's sake we specialcase this indiscriminately.
    if n_model1_correct == n_model2_correct:
        return 0.0, 1.0

    return statsmodels_ztest(
        count=[n_model1_correct, n_model2_correct],
        nobs=[test_set_size, test_set_size],
        alternative="two-sided",
    )


def infer_complete_contingency_table(
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
        absolute_overlap: int,
) -> Sequence[Sequence[int]]:
    """
    Infer a complete McNemar/correctness contingency table given the margins and the absolute overlap.

    Note that the absolute overlap must be consistent with the other values passed. If it is not consistent we will
    raise a ValueError. For the overlap to be consistent, its parity must match that of the minimum possible
    overlap.

    The easiest way to force your absolute overlap's parity to match is to add to it:

        fixed_absolute_overlap = (absolute_overlap % 2) - (min_possible_overlap % 2)
    """
    # Make sure that the absolute overlap has the same parity as the minimum possible overlap.
    # We check this here because otherwise we will end up with inconsistent entries and a misleading error.
    twice_n_both_correct = (absolute_overlap + n_model1_correct + n_model2_correct - test_set_size)
    if twice_n_both_correct % 2 != 0:
        raise ValueError(
            "The absolute overlap value passed cannot lead to a consistent contingency table. "
            "Make sure the parity of the absolute overlap matches that of a known-consistent overlap value "
            "(such as the minimum possible absolute overlap)."
        )

    n_both_correct = twice_n_both_correct // 2
    n_with_exactly_one_model_correct = test_set_size - absolute_overlap
    n_only_model1_correct = n_model1_correct - n_both_correct
    n_only_model2_correct = n_model2_correct - n_both_correct
    n_both_wrong = absolute_overlap - n_both_correct
    if n_both_correct + n_only_model1_correct + n_only_model2_correct + n_both_wrong != test_set_size:
        raise RuntimeError(
            "Inferred contingency table is not consistent with test set size."
        )
    if n_only_model1_correct + n_only_model2_correct != n_with_exactly_one_model_correct:
        raise RuntimeError(
            "Inferred contingency table is not consistent with number of disagreements."
        )
    if n_only_model1_correct + n_both_correct != n_model1_correct:
        raise RuntimeError(
            "Inferred contingency table is not consistent with number of questions model 1 answered correctly."
        )
    if n_only_model2_correct + n_both_correct != n_model2_correct:
        raise RuntimeError(
            "Inferred contingency table is not consistent with number of questions model 2 answered correctly."
        )

    # Contingency table where the rows cover model 1 and columns cover model 2
    return [
        [n_both_correct, n_only_model1_correct],
        [n_only_model2_correct, n_both_wrong],
    ]


def statsmodels_mcnemar(
        contingency_table: Sequence[Sequence[int]],
) -> Tuple[float, float]:
    """
    Compare models using the Statsmodels version of the chi-squared McNemar test.
    """
    result = _statsmodels_mcnemar(contingency_table, exact=False, correction=False)
    return result.statistic, result.pvalue


def statsmodels_mcnemar_exact(
        contingency_table: Sequence[Sequence[int]],
) -> Tuple[float, float]:
    """
    Compare models using the Statsmodels version of the exact (i.e. binomial) McNemar test.
    """
    result = _statsmodels_mcnemar(contingency_table, exact=True, correction=False)
    return result.statistic, result.pvalue


def mcnemar_mid_p(
        *,
        n_model1_correct: int,
        n_model2_correct: int,
        absolute_overlap: int,
        test_set_size: int
) -> Tuple[float, float]:
    """
    Compare the two models whose accuracy is given using the mid-p version of McNemar's test and return the test
    statistic and p-value.

    We do this by setting up a contingency table like this:

    |   rows: model 1 / columns: model 2 | M2 correct | M2 incorrect |
    | ---------------------------------- | ---------- | ------------ |
    |                         M1 correct |          a |            b |
    |                       M1 incorrect |          c |            d |

    where a, b, c, and d are integers.

    In particular

    test_set_size = N = a + b + c + d,
    model1_accuracy = (a + b)/N,
    model2_accuracy = (a + c)/N,
    percent_overlap = (a + d)/N,

    which we can use to calculate b and b + c. The mid-p McNemar test calculates the p-value under the assumption that
    under the null hypothesis, and when b >= c, b has a binomial distribution with n = b + c and p = 0.5. (When b < c we
    calculate the p-value using c instead.)

    To calculate the mid-p variant, we calculate the two-tailed probability for the observed value and then
    subtract the probability of the observed value. (Equivalently, we calculate the one-tailed probability, subtract
    half the probability of the observed value, and double the result to get a two-tailed probability.)
    """
    contingency_table = infer_complete_contingency_table(
        test_set_size, n_model1_correct, n_model2_correct, absolute_overlap
    )
    result = _statsmodels_mcnemar(contingency_table, exact=True, correction=False)

    observation = max(contingency_table[0][1], contingency_table[1][0])
    n_disagreements = contingency_table[0][1] + contingency_table[1][0]
    return result.statistic, (
            result.pvalue - binom.pmf(k=observation, n=n_disagreements, p=0.5)
    )


def get_min_possible_absolute_overlap(test_set_size: int, model1_correct: int, model2_correct: int) -> int:
    """
    Return the minimum possible absolute overlap between models 1 and 2.
    """
    return max(model1_correct + model2_correct - test_set_size, 0) + max(test_set_size - model1_correct - model2_correct, 0)


def get_max_possible_absolute_overlap(test_set_size: int, model1_correct: int, model2_correct: int) -> int:
    """
    Return the maximum possible absolute overlap between models 1 and 2.
    """
    return test_set_size - abs(model2_correct - model1_correct)


def mcnemar_min_overlap(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Return an upper bound on the McNemar's test p-value by bounding the denominator (the number of discordant examples)
    from above.

    This results in a bound on the test statistic from below.

    That is, we perform the test as if the two models disagreed on every question possible, considering their observed
    accuracies. This maximizes the variance in the McNemar's test statistic.

    This returns a pair (chi-squared, p-value).
    """
    absolute_overlap = get_max_possible_absolute_overlap(
        test_set_size,
        n_model1_correct,
        n_model2_correct,
    )
    # statsmodels does not handle the edge case where the models overlap everywhere (it raises an exception),
    # so we have to handle it ourselves.
    if absolute_overlap == test_set_size:
        return 0., 1.
    contingency_table = infer_complete_contingency_table(test_set_size, n_model1_correct, n_model2_correct, absolute_overlap)
    return statsmodels_mcnemar(contingency_table)


def mcnemar_max_overlap(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Return a lower bound on the McNemar's test p-value by bounding the denominator (the number of discordant examples)
    from below.

    This results in a bound on the test statistic from above.

    That is, we perform the test as if the two models agreed on every question possible, considering their observed
    accuracies. This minimizes the variance in the McNemar's test statistic

    This returns a pair (chi-squared, p-value).
    """
    absolute_overlap = get_max_possible_absolute_overlap(
        test_set_size,
        n_model1_correct,
        n_model2_correct,
    )
    # statsmodels does not handle the edge case where the models overlap everywhere (it raises an exception),
    # so we have to handle it ourselves.
    if absolute_overlap == test_set_size:
        return 0., 1.
    contingency_table = infer_complete_contingency_table(test_set_size, n_model1_correct, n_model2_correct, absolute_overlap)
    return statsmodels_mcnemar(contingency_table)


def _all_possible_mcnemar_exact_results(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> List[Tuple[float, float]]:
    """
    Return a list of all possible McNemar exact test results for the given pair
    """
    min_overlap = get_min_possible_absolute_overlap(test_set_size, n_model1_correct, n_model2_correct)
    max_overlap = get_max_possible_absolute_overlap(test_set_size, n_model1_correct, n_model2_correct)

    test_results = []
    for absolute_overlap in range(min_overlap, max_overlap + 1, 2):
        contingency_table = infer_complete_contingency_table(
            test_set_size, n_model1_correct, n_model2_correct, absolute_overlap
        )
        test_results.append(
            statsmodels_mcnemar_exact(contingency_table)
        )
    return test_results


def mcnemar_exact_lower_p(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Return a lower bound on the exact conditional McNemar's test p-value.
    """
    possible_results = _all_possible_mcnemar_exact_results(
        test_set_size=test_set_size,
        n_model1_correct=n_model1_correct,
        n_model2_correct=n_model2_correct,
    )
    return min(possible_results, key=lambda t: t[1])


def mcnemar_exact_upper_p(
        *,
        test_set_size: int,
        n_model1_correct: int,
        n_model2_correct: int,
) -> Tuple[float, float]:
    """
    Return an upper bound on the exact conditional McNemar's test p-value.
    """
    possible_results = _all_possible_mcnemar_exact_results(
        test_set_size=test_set_size,
        n_model1_correct=n_model1_correct,
        n_model2_correct=n_model2_correct,
    )
    return max(possible_results, key=lambda t: t[1])
