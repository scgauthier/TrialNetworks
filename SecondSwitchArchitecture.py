import numpy as np
from scipy.stats import bernoulli
from scipy.special import binom
from random import sample, choice, shuffle, random
from math import floor
from typing import Tuple


# Graph useful b/c any pairing from N nodes can be re-written as one of
# the edges of this graph, so indices of Tuple of graph edges correspond to
# unique identifiers for the VOQs.
def gen_VOQ_graph(NumUsers: int) -> Tuple:
    Network_graph = []

    for requester in range(NumUsers-1):
        for partner in range(requester, NumUsers-1):
            Network_graph.append((requester, partner))

    return Network_graph


def form_ordered_pairs(NumUsers: int) -> Tuple:

    pairs = []
    senders = sample(range(NumUsers), floor(NumUsers / 2))
    reciever_pool = list(range(NumUsers))
    for sender in senders:
        reciever_pool.remove(sender)

    if (NumUsers % 2) != 0:
        reciever_pool.remove(choice(reciever_pool))
    shuffle(reciever_pool)
    for x in range(floor(NumUsers / 2)):
        if senders[x] < reciever_pool[x]:
            pairs.append((senders[x], reciever_pool[x] - 1))
        else:
            pairs.append((reciever_pool[x], senders[x] - 1))

    return pairs


def gen_uniform_arrivals(NumUsers: int, rate: float) -> np.ndarray:

    pairs = form_ordered_pairs(NumUsers)
    VOQs = gen_VOQ_graph(NumUsers)
    arrivals = np.zeros((len(VOQs), 1))

    for pair in pairs:
        ind = VOQs.index(pair)
        arrivals[ind, 0] = bernoulli.rvs(rate, size=1)[0]

    return arrivals


def fact(Number: int) -> int:
    factorial = 1
    for x in range(1, Number + 1):
        factorial *= x
    return factorial


def calc_total_matchings(NumUsers: int, H_max: int) -> int:
    total_matchings = 1
    for x in range(H_max):
        total_matchings *= binom((NumUsers - (2 * x)), 2)
    total_matchings *= (1 / fact(H_max))
    return int(total_matchings)


def generate_size_2_matchings(NumUsers: int) -> Tuple:

    matchings = []
    for x in range(NumUsers - 2):
        for y in range(x + 1, NumUsers):
            options = list(range(x + 1, NumUsers))
            options.remove(y)
            if options:
                for u in range(len(options)):
                    for v in range(u + 1, len(options)):
                        matchings.append([(x, y - 1),
                                         (options[u], options[v] - 1)])

    return matchings


def determine_matched_VOQs(NumUsers: int) -> Tuple:

    VOQ_matchings = []
    matchings = generate_size_2_matchings(NumUsers)
    graph = gen_VOQ_graph(NumUsers)

    for matching in matchings:
        VOQs = []
        for edge in matching:
            VOQs.append(graph.index(edge))
        VOQ_matchings.append(VOQs)

    return VOQ_matchings


def select_max_weight_matching(NumUsers: int, possible_matchings: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
    max_weight = 0

    for matching in possible_matchings:
        weight = 0
        array_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
        for VOQ_ind in matching:
            weight += current_queue_lengths[VOQ_ind]
            array_matching[VOQ_ind, 0] = 1

        if weight > max_weight:
            max_weight = weight
            max_matching = array_matching
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_matching = array_matching
    return max_matching


def simulate_queue_lengths(NumUsers: int,
                           H_num: int,
                           rate: float,
                           iters: int,
                           sampling: bool) -> Tuple[np.ndarray, np.ndarray]:
    queue_lengths = np.zeros((sum(range(NumUsers)), iters))
    ql = np.zeros(iters)

    possible_matchings = []
    for x in range(len(gen_VOQ_graph(NumUsers))):
        possible_matchings.append([x])
    if H_num > 1:
        for matching in determine_matched_VOQs(NumUsers):
            possible_matchings.append(matching)
    # if H_num == 3:
    #     for matching in define_size_3_matchings(NumUsers):
    #         possible_matchings.append(matching)

    for x in range(iters):
        arrivals = gen_uniform_arrivals(NumUsers, rate)
        matching = np.zeros((int(NumUsers * (NumUsers - 1) / 2), 1))
        if x > 0:
            matching = select_max_weight_matching(NumUsers,
                                                  possible_matchings,
                                                  queue_lengths[:, x - 1])
        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:, 0]
                               - matching[:, 0])
        for j in range(int(NumUsers * (NumUsers - 1) / 2)):
            if queue_lengths[j, x] < 0:
                queue_lengths[j, x] = 0
        ql[x] = np.sum(queue_lengths[:, x], axis=0)

    samples = []
    if sampling:
        for x in np.linspace(0, iters-1, 10):
            samples.append(ql[int(x)])

    return samples, ql


print(simulate_queue_lengths(6, 2, 0.6, 1000, True)[0])
