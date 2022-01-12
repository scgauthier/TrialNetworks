import numpy as np
from scipy.special import binom as bc
from scipy.stats import norm
from random import random


def F1_rate_declare(change_rate: float,
                    mean_rate: float,
                    rate_var: float,
                    rates: np.ndarray) -> np.ndarray:

    NQs = np.shape(rates)[0]
    up_rates = np.copy(rates)
    for x in range(NQs):
        if random() < (change_rate / NQs):  # Change rate equally distributed
            up_rates[x, 0] = norm.rvs(mean_rate, rate_var, size=1)[0]
    return up_rates


# How will continuous time simulation work?
# Run for time x \in [0, Stop)

# At each x check if any pair wants to alter rate declaration
# Decide if rate changes randomly (random number < change rate)

# Rate declare drawn from gaussian dist with mean less than the
# expected single share of the threshold, variance of distance from mean to
# expected single share

# If one or more rate declarations changed, wait time x = tau before
# re-scheduling

# Gen pairs for all scheduled: Bernoulli process with mean p_succ
# Update recorded rate every time a pair successfully created
