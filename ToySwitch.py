# import numpy as np
from scipy.stats import bernoulli
from typing import Tuple


# Assume lambda12=lambda21, 13=31, 23=32
def generate_arivals(lambda12: float, lambda13: float,
                     lambda23: float) -> Tuple[int, int, int, int, int, int]:

    a12 = bernoulli.rvs(lambda12, size=1)[0]
    a13 = bernoulli.rvs(lambda13, size=1)[0]
    a23 = bernoulli.rvs(lambda23, size=1)[0]

#    #Encode that only 1 of each pair submits request per slot
    if a12 != 1:
        a21 = bernoulli.rvs(lambda12, size=1)[0]
    else:
        a21 = 0

    if a13 != 1:
        a31 = bernoulli.rvs(lambda13, size=1)[0]
    else:
        a31 = 0

    if a23 != 1:
        a32 = bernoulli.rvs(lambda23, size=1)[0]
    else:
        a32 = 0

    return a12, a13, a21, a23, a31, a32


# Network_graph = np.array([[0, 1, 1], [1, 0, 1], [0, 1, 1]])

Matchings = {
    'M1': [[1, 2], [2, 3], [3, 1]],
    'M2': [[1, 2], [2, 1]],
    'M3': [[1, 3], [2, 1], [3, 2]],
    'M4': [[1, 3], [3, 1]],
    'M5': [[2, 3], [3, 2]]
}
