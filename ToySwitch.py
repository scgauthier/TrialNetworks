import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import binom
from typing import Tuple, List
from random import random
from itertools import combinations, combinations_with_replacement
from matplotlib import rc

rc('text', usetex=True)


# Assume arrivals are binomial random variables, such that up to a max of m
# requests submitted per VOQ per slot. Probability of each request submission
# for queue i is p_i. Distribution of p_i's fixed in main code.
def gen_arivals(max_submissions: int,
                probs: List[float],
                NumUsers: int) -> np.ndarray:

    cutoff = int(NumUsers * (NumUsers - 1) / 2)
    arrivals = np.zeros((cutoff, 1))

    for x in range(cutoff):
        arrivals[x, 0] = binom.rvs(max_submissions, probs[x], size=1)[0]

    return arrivals

# print(generate_arivals((0.25,0.25,0.25,0.25,0.25,0.25)))


# Here network graph just corresponds to all ordered pairs of nodes, i.e.
# the distinct VOQs
def gen_network_graph(NumUsers: int) -> Tuple:
    Network_graph = []

    for requester in range(NumUsers-1):
        for partner in range(requester, NumUsers-1):
            Network_graph.append((requester, partner))

    return Network_graph


# All distinct ways of choosing h different VOQs
def get_size_h_matchings(h: int, NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations(range(pool_size), h))


# In case want to be able to schedule one VOQ more than once per slot
def get_size_h_flexible_matchings(h: int,
                                  NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations_with_replacement(range(pool_size), h))


def select_max_weight_matching(possible_matchings: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
    max_weight = 0

    for matching in possible_matchings:
        weight = 0
        array_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
        for edge in matching:
            weight += current_queue_lengths[edge]
            array_matching[edge, 0] = 1

        if weight > max_weight:
            max_weight = weight
            max_matching = array_matching
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_matching = array_matching
    return max_matching


def simulate_queue_lengths(NumUsers: int, H_num: int,
                           max_subs: int,
                           pDist: str, prob_param: float,
                           iters: int) -> Tuple[np.ndarray, np.ndarray]:
    queue_lengths = np.zeros((sum(range(NumUsers)), iters))
    ql = np.zeros(iters)

    possible_matchings = []
    for x in range(H_num + 1):
        possible_matchings += get_size_h_matchings(x, NumUsers)

    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        probs = [prob_param] * int(bc(NumUsers, 2))

    else:
        probs = [0] * int(bc(NumUsers, 2))

    for x in range(iters):
        arrivals = gen_arivals(max_subs, probs, NumUsers)
        matching = np.zeros((int(NumUsers * (NumUsers - 1) / 2), 1))
        if x > 0:
            matching = select_max_weight_matching(possible_matchings,
                                                  queue_lengths[:, x - 1])
        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:, 0]
                               - matching[:, 0])
        for j in range(int(NumUsers * (NumUsers - 1) / 2)):
            if queue_lengths[j, x] < 0:
                queue_lengths[j, x] = 0
        ql[x] = np.sum(queue_lengths[:, x], axis=0)

    return ql


# rate = 0.15
# iters = 1000
# rates = (rate, rate, rate, rate, rate, rate)
# final_slice, ql = simulate_queue_lengths(4, 1, rates, iters)
# final_slice2, ql2 = simulate_queue_lengths(4, 2, rates, iters)
# final_slice3, ql3 = simulate_queue_lengths(4, 3, rates, iters)
#
# plt.plot(range(iters), ql)
# plt.plot(range(iters), ql2)
# plt.plot(range(iters), ql3)
# plt.show()
# possible_matchings = []
# for edge in gen_network_graph(4):
#     possible_matchings.append([edge])
# queues = np.array([[3], [2], [2], [4], [8], [1]])
# matching = select_max_weight_matching(possible_matchings, queues)
# print(matching)
# size = 4
# print('size 1: ', len(gen_network_graph(size)))
# print('size 2: ')
# print(len(define_size_2_matchings(size)), '\n\n',
#       define_size_2_matchings(size), '\n\n')
# print('size 3: ')
# print(len(define_size_3_matchings(size)), '\n\n',
#       define_size_3_matchings(size))
