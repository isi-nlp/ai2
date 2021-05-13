# jac: Copied from myself. Source of code:
#      the module isi_power_analyses.common
#      in github.com/isi-vista/common-sense-power-analysis
from typing import Tuple

import numpy as np
import scipy.stats
from scipy.stats import chi2


def fishers_test(
    *,
    test_set_size: int,
    model1_accuracy: float,
    model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Compute the significance (p-value) for a comparison between the given two models using Fisher's exact test.

    We do this by setting up a contingency table like this:

    | rows: correctness / columns: model | model 1 | model 2 |
    | ---------------------------------- | ------- | ------- |
    |                            correct |       a |       b |
    |                          incorrect |       c |       d |

    Note that this is *very different* from the contingency table we use for McNemar's test in other parts of the code.

    Note also that this specifically performs the *two-tailed* version of Fisher's exact test.

    This returns a pair (odds ratio, p-value).
    """
    model1_correct = int(model1_accuracy * test_set_size)
    model2_correct = int(model2_accuracy * test_set_size)
    return scipy.stats.fisher_exact(
        [
            [model1_correct, model2_correct],
            [test_set_size - model1_correct, test_set_size - model2_correct],
        ],
        # jac: https://statisticsbyjim.com/hypothesis-testing/use-one-tailed-tests/
        # makes a persuasive case that we really want a two-sided hypothesis test, not a one-sided one
        alternative="two-sided",
    )


def binomial_difference_of_proportions_test(
        *,
        test_set_size: int,
        model1_accuracy: float,
        model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Compute the significance (p-value) for a comparison between the given two models using a binomial test
    for a difference of proportions.

    We do this by setting up a contingency table like this:

    | rows: correctness / columns: model | model 1 | model 2 |
    | ---------------------------------- | ------- | ------- |
    |                            correct |       a |       b |
    |                          incorrect |       c |       d |

    Note that this is *very different* from the contingency table we use for McNemar's test in other parts of the code.

    Note also that this specifically performs the *two-tailed* version of the difference of proportions test.

    This returns a pair (z-value, p-value).
    """
    # Handle edge case where models have identical accuracy
    # In this case the test statistic should be 0 (inasmuch as it make sense)
    # and the p-value should be 1
    if model1_accuracy == model2_accuracy:
        return 0.0, 1.0

    # This test calculation is based on https://www.statology.org/two-proportion-z-test/
    # statsmodels.stats performs this test similarly:
    # https://www.statsmodels.org/v0.10.2/_modules/statsmodels/stats/proportion.html#proportions_ztest
    # jac: should try and confirm this formula using Wikipedia
    # Because both models are run on the same test set, we can factor out n_1 = n_2 = test_set_size in the formula.
    pooled_accuracy = (model1_accuracy + model2_accuracy) / 2.

    # Whether we compute a_1 - a_2 or a_2 - a_1 doesn't change the p-value because the distribution
    # used for this test is normal (thus symmetric).
    # We arbitrarily use model2_accuracy - model1_accuracy.
    #
    # Also, (1/n_1 + 1/n_2) = 2 / n because n_1 = n_2 = test_set_size (aka n).
    test_statistic = (model2_accuracy - model1_accuracy) / np.sqrt(
        pooled_accuracy * (1. - pooled_accuracy) * 2. / test_set_size
    )

    # This is a two-tailed test, and the normal distribution is symmetric, so we multiply by 2 to get the probability
    # in the "other tail."
    #
    # We use the absolute value of test_statistic so that the survival function will correctly return
    # the tail probability when model2_accuracy - model1_accuracy is negative.
    return test_statistic, 2. * scipy.stats.norm.sf(np.abs(test_statistic))


def mcnemar(
    *,
    model1_accuracy: float,
    model2_accuracy: float,
    fractional_agreement: float,
    test_set_size: int
) -> Tuple[float, float]:
    """
    Compare the two models whose accuracy is given using McNemar's test and return the test statistic and p-value.
    """
    accuracy_difference = model2_accuracy - model1_accuracy
    test_statistic = test_set_size * accuracy_difference * accuracy_difference / (1 - fractional_agreement)
    return test_statistic, chi2.sf(test_statistic, 1)


def get_min_possible_agreement(model1_accuracy: float, model2_accuracy: float) -> float:
    """
    Return the minimum possible agreement between models 1 and 2.
    """
    return max(model1_accuracy + model2_accuracy - 1., 0.)


def get_max_possible_agreement(model1_accuracy: float, model2_accuracy: float) -> float:
    """
    Return the maximum possible agreement between models 1 and 2.
    """
    return 1 - np.abs(model2_accuracy - model1_accuracy)


def mcnemar_worst_case(
        *,
        test_set_size: int,
        model1_accuracy: float,
        model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Return an upper bound on the McNemar's test p-value by bounding the denominator (the number of discordant examples)
    from above.

    This results in a bound on the test statistic from below.

    That is, we perform the test as if the two models disagreed on every question possible, considering their observed
    accuracies. This maximizes the variance in the McNemar's test statistic.

    This returns a pair (chi-squared, p-value).
    """
    # Handle edge case where models have identical accuracy
    # In this case the test statistic should be 0 (inasmuch as it make sense)
    # and the p-value should be 1
    if model1_accuracy == model2_accuracy:
        return 0.0, 1.0

    # Let M = max(model1_accuracy + model2_accuracy - 1., 0.).
    # M is a sharp lower bound on P_a.
    # Let d = 1 - M. Then:
    #
    # d = 1 - max(model1_accuracy + model2_accuracy - 1., 0.)
    #   = 1. + min(1. - model1_accuracy - model2_accuracy, 0.)
    #   = min(2. - model1_accuracy - model2_accuracy, 1.)
    #
    # The fractional discordance 1 - P_a is bounded sharply above by d,
    # hence 1/(1 - P_a) is bounded below by 1/d,
    # hence the McNemar's test statistic (test_set_size * (delta_acc ** 2) / (1 - P_a)
    # is bounded above by test_set_size * (delta_acc ** 2) / d
    delta_acc = (model2_accuracy - model1_accuracy)
    # discordance_upper_bound = 1 - max(model1_accuracy + model2_accuracy - 1., 0.)
    discordance_upper_bound = 1 - get_min_possible_agreement(model1_accuracy, model2_accuracy)
    test_statistic = test_set_size * delta_acc * delta_acc / discordance_upper_bound
    return test_statistic, chi2.sf(test_statistic, 1)


def mcnemar_best_case(
        *,
        test_set_size: int,
        model1_accuracy: float,
        model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Return a lower bound on the McNemar's test p-value by bounding the denominator (the number of discordant examples)
    from below.

    This results in a bound on the test statistic from above.

    That is, we perform the test as if the two models agreed on every question possible, considering their observed
    accuracies. This minimizes the variance in the McNemar's test statistic

    This returns a pair (chi-squared, p-value).
    """
    # The fractional discordance 1 - P_a is bounded sharply below by |delta_acc|,
    delta_acc = (model2_accuracy - model1_accuracy)
    test_statistic = test_set_size * abs(delta_acc)
    return test_statistic, chi2.sf(test_statistic, 1)
