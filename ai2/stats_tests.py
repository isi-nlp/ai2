# jac: Copied from myself. Source of code:
#      the module isi_power_analyses.common
#      in github.com/isi-vista/common-sense-power-analysis
import logging
from math import ceil, floor
from typing import List, Tuple

from attr import attrs, attrib
from attr.validators import instance_of, and_, in_
import numpy as np
import scipy.stats
from scipy.stats import chi2, binom_test, binom
from vistautils.range import Range


logger = logging.getLogger(__name__)


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
    percent_overlap: float,
    test_set_size: int
) -> Tuple[float, float]:
    """
    Compare the two models whose accuracy is given using McNemar's test and return the test statistic and p-value.
    """
    accuracy_difference = model2_accuracy - model1_accuracy
    test_statistic = test_set_size * accuracy_difference * accuracy_difference / (1 - percent_overlap)
    return test_statistic, chi2.sf(test_statistic, 1)


@attrs
class _BinomialMcNemarTableInfo:
    """
    Holds the table info needed to calculate the exact conditional McNemar test.
    """
    n_disagreements: int = attrib(validator=and_(instance_of(int), in_(Range.at_least(0))))
    n_only_model1_correct: int = attrib(validator=and_(instance_of(int), in_(Range.at_least(0))))
    n_only_model2_correct: int = attrib(validator=and_(instance_of(int), in_(Range.at_least(0))))


def _approx_disagreements(
    test_set_size: int,
    percent_overlap: float,
) -> float:
    """
    Return the number of disagreements, estimated by test_set_size * (1 - percent_overlap).

    We leave this as a fractional value so that you can round it however you want.
    """
    return test_set_size * (1 - percent_overlap)


def _get_n_only_model1_correct(
    test_set_size: int,
    n_disagreements: int,
    model1_accuracy: float,
    model2_accuracy: float,
) -> int:
    """
    Infer the number of questions which only model1 got correct from the test set size, number of disagreements, and
    model accuracies.
    """
    return round(
        (
                n_disagreements - test_set_size * (model1_accuracy - model2_accuracy)
        ) / 2
    ) if n_disagreements > 0 else 0


def _infer_mcnemar_table_info(
        *,
        model1_accuracy: float,
        model2_accuracy: float,
        percent_overlap: float,
        test_set_size: int
) -> _BinomialMcNemarTableInfo:
    """
    Infer the relevant table info for computing the binomial McNemar test given the test set size, number of
    disagreements, and model accuracies.
    """
    # n_disagreements = b + c = test_set_size * (1 - percent_overlap)
    # b - c = test_set_size * (model1_accuracy - model2_accuracy)
    # so b = (1/2)(n_disagreements - test_set_size * (model1_accuracy - model2_accuracy))
    #
    # We get our accuracies as floats, so we can't just multiply out and get integers.
    # We have to round.
    n_disagreements = round(_approx_disagreements(test_set_size, percent_overlap))
    n_only_model1_correct = _get_n_only_model1_correct(
        test_set_size, n_disagreements, model1_accuracy, model2_accuracy
    )
    n_only_model2_correct = n_disagreements - n_only_model1_correct

    return _BinomialMcNemarTableInfo(n_disagreements, n_only_model1_correct, n_only_model2_correct)


def _mcnemar_exact_conditional_from_table_info(
    table_info: _BinomialMcNemarTableInfo,
) -> Tuple[float, float]:
    """
    Calculate the exact conditional (i.e. binomial) McNemar test using the relevant table info.
    """
    if table_info.n_disagreements > 0:
        x = max(table_info.n_only_model1_correct, table_info.n_only_model2_correct)
        test_statistic = x / table_info.n_disagreements
        return test_statistic, binom_test(x=x, n=table_info.n_disagreements, p=0.5, alternative="two-sided")
    else:
        return 0, 1.


def mcnemar_exact(
    *,
    model1_accuracy: float,
    model2_accuracy: float,
    percent_overlap: float,
    test_set_size: int
) -> Tuple[float, float]:
    """
    Compare the two models whose accuracy is given using the exact version of McNemar's test and return the test
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

    which we can use to calculate b and b + c. The exact McNemar test calculates the p-value under the assumption that
    under the null hypothesis, and when b >= c, b has a binomial distribution with n = b + c and p = 0.5. (When b < c we
    calculate the p-value using c instead.)

    Thanks to a reviewer who pointed out this version of McNemar's test.
    """
    table_info = _infer_mcnemar_table_info(
        model1_accuracy=model1_accuracy,
        model2_accuracy=model2_accuracy,
        percent_overlap=percent_overlap,
        test_set_size=test_set_size,
    )
    return _mcnemar_exact_conditional_from_table_info(table_info)


def mcnemar_mid_p(
        *,
        model1_accuracy: float,
        model2_accuracy: float,
        percent_overlap: float,
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
    # n_disagreements = b + c = test_set_size * (1 - percent_overlap)
    # b - c = test_set_size * (model1_accuracy - model2_accuracy)
    # so b = (1/2)(n_disagreements - test_set_size * (model1_accuracy - model2_accuracy))
    #
    # We get our accuracies as floats, so we can't just multiply out and get integers.
    # We have to round.
    table_info = _infer_mcnemar_table_info(
        model1_accuracy=model1_accuracy,
        model2_accuracy=model2_accuracy,
        percent_overlap=percent_overlap,
        test_set_size=test_set_size,
    )
    x = max(table_info.n_only_model1_correct, table_info.n_only_model2_correct)
    test_statistic = x / table_info.n_disagreements
    return test_statistic, (
       binom_test(x=x, n=table_info.n_disagreements, p=0.5, alternative="two-sided")
       - binom.pmf(k=x, n=table_info.n_disagreements, p=0.5)
    )


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


def mcnemar_min_overlap(
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


def mcnemar_max_overlap(
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


def _all_possible_mcnemar_exact_results(
    *,
    test_set_size: int,
    model1_accuracy: float,
    model2_accuracy: float,
) -> List[Tuple[float, float]]:
    """
    Return a list of all possible McNemar exact test results for the given pair
    """
    min_disagreements = floor(
        _approx_disagreements(
            test_set_size,
            get_max_possible_agreement(
                model1_accuracy=model1_accuracy,
                model2_accuracy=model2_accuracy,
            ),
        )
    )
    if min_disagreements % 2 != test_set_size % 2:
        min_disagreements -= 1
    max_disagreements = ceil(
        _approx_disagreements(
            test_set_size,
            get_min_possible_agreement(
                model1_accuracy=model1_accuracy,
                model2_accuracy=model2_accuracy,
            ),
        )
    )
    if max_disagreements % 2 != test_set_size % 2:
        max_disagreements += 1

    test_results = []
    for n_disagreements in range(min_disagreements, max_disagreements, 2):
        n_only_model1_correct = _get_n_only_model1_correct(
            test_set_size, n_disagreements, model1_accuracy, model2_accuracy
        )
        n_only_model2_correct = n_disagreements - n_only_model1_correct
        if n_only_model1_correct >= 0 and n_only_model2_correct >= 0:
            test_results.append(
                _mcnemar_exact_conditional_from_table_info(
                    _BinomialMcNemarTableInfo(
                        n_disagreements=n_disagreements,
                        n_only_model1_correct=n_only_model1_correct,
                        n_only_model2_correct=n_only_model2_correct,
                    )
                )
            )
        else:
            logger.debug(
                "Skipping exact conditional McNemar test for impossible absolute disagreement value %d.",
                n_disagreements,
            )
    return test_results


def mcnemar_exact_lower_p(
    *,
    test_set_size: int,
    model1_accuracy: float,
    model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Return a lower bound on the exact conditional McNemar's test p-value.
    """
    possible_results = _all_possible_mcnemar_exact_results(
        test_set_size=test_set_size,
        model1_accuracy=model1_accuracy,
        model2_accuracy=model2_accuracy,
    )
    return min(possible_results, key=lambda t: t[1])


def mcnemar_exact_upper_p(
        *,
        test_set_size: int,
        model1_accuracy: float,
        model2_accuracy: float,
) -> Tuple[float, float]:
    """
    Return an upper bound on the exact conditional McNemar's test p-value.
    """
    possible_results = _all_possible_mcnemar_exact_results(
        test_set_size=test_set_size,
        model1_accuracy=model1_accuracy,
        model2_accuracy=model2_accuracy,
    )
    return max(possible_results, key=lambda t: t[1])
