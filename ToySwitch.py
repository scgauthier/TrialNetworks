import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from typing import Tuple
from random import random


# Assume lambda12=lambda21, 13=31, 14=41, 23=32, 24=42, 34=43
def generate_uniform_arivals(rate: float, NumUsers: int) -> np.ndarray:

    cutoff = int(NumUsers * (NumUsers - 1) / 2)
    arrivals = np.zeros((cutoff, 1))

    for x in range(cutoff):
        arrivals[x, 0] = bernoulli.rvs(rate, size=1)[0]

    return arrivals

# print(generate_arivals((0.25,0.25,0.25,0.25,0.25,0.25)))


def gen_network_graph(NumUsers: int) -> Tuple:
    Network_graph = []

    for requester in range(NumUsers-1):
        for partner in range(requester, NumUsers-1):
            Network_graph.append((requester, partner))

    return Network_graph


# Function defines matchings by iterating through all possible first edges in
# a matching, removing all edges shared by one of the two nodes in the first
# edge, and finally going systematically through each of the other edges in the
# graph and choosing them as a second edge for the matching
# I think iterating through the first edge in this way helps prevent double
#  counting?
def define_size_2_matchings(NumUsers: int) -> Tuple:
    matchings = []
    Network_graph = gen_network_graph(NumUsers)

    for x in range(NumUsers - 1):
        local_graph = Network_graph.copy()
        for w in range(NumUsers - 1):
            try:
                start = local_graph.index((x, w))
                local_graph = local_graph[(start + 1):]
            except ValueError:
                continue

        for y in range(x, NumUsers - 1):
            edge1 = (x, y)
            sub_local_graph = local_graph.copy()

            for w in range(NumUsers - 1):
                try:
                    sub_local_graph.remove((w, y))
                except ValueError:
                    continue

            for z in range(len(sub_local_graph)):
                matchings.append([edge1, sub_local_graph[z]])

            sub_local_graph.clear()
        local_graph.clear()

    return matchings


def define_size_3_matchings(NumUsers: int) -> Tuple:
    matchings = []
    Network_graph = gen_network_graph(NumUsers)

    for x in range(NumUsers - 1):
        local_graph = Network_graph.copy()
        for w in range(NumUsers - 1):
            try:
                start = local_graph.index((x, w))
                local_graph = local_graph[(start + 1):]
            except ValueError:
                pass

        for y in range(x, NumUsers - 1):
            edge1 = (x, y)
            sub_local_graph = local_graph.copy()

            for w in range(NumUsers - 1):
                try:
                    sub_local_graph.remove((w, y))
                except ValueError:
                    pass

            for z in range(len(sub_local_graph)):
                edge2 = sub_local_graph[z]
                sub_sub_local_graph = sub_local_graph.copy()
                sub_sub_local_graph = sub_sub_local_graph[(z+1):]
                if len(sub_sub_local_graph) > 0:

                    for w in range(NumUsers-1):
                        try:
                            sub_sub_local_graph.remove((edge2[0], w))
                        except ValueError:
                            pass
                        try:
                            sub_sub_local_graph.remove((edge2[1], w))
                        except ValueError:
                            pass
                        try:
                            sub_sub_local_graph.remove((w, edge2[0]))
                        except ValueError:
                            pass
                        try:
                            sub_sub_local_graph.remove((w, edge2[1]))
                        except ValueError:
                            pass
                    for a in range(len(sub_sub_local_graph)):
                        matchings.append([edge1, edge2,
                                         sub_sub_local_graph[a]])

                sub_sub_local_graph.clear()
            sub_local_graph.clear()
        local_graph.clear()
    return matchings


def select_max_weight_matching(possible_matchings: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
    max_weight = 0
    graph = gen_network_graph(np.shape(current_queue_lengths)[0] - 2)

    for matching in possible_matchings:
        weight = 0
        array_matching = np.zeros((np.shape(current_queue_lengths)[0], 1))
        for edge in matching:
            ind = graph.index(edge)
            weight += current_queue_lengths[ind]
            array_matching[ind, 0] = 1
            # if (edge[0] == 0):
            #     if (edge[1] == 0):
            #         weight += current_queue_lengths[0]
            #         array_matching[0, 0] = 1
            #     elif (edge[1] == 1):
            #         weight += current_queue_lengths[1]
            #         array_matching[1, 0] = 1
            #     elif (edge[1] == 2):
            #         weight += current_queue_lengths[2]
            #         array_matching[2, 0] = 1
            # elif (edge[0] == 1):
            #     if (edge[1] == 1):
            #         weight += current_queue_lengths[3]
            #         array_matching[3, 0] = 1
            #     elif (edge[1] == 2):
            #         weight += current_queue_lengths[4]
            #         array_matching[4, 0] = 1
            # else:
            #     weight += current_queue_lengths[5]
            #     array_matching[5, 0] = 1
        if weight > max_weight:
            max_weight = weight
            max_matching = array_matching
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_matching = array_matching
    return max_matching


def simulate_queue_lengths(NumUsers: int, H_num: int,
                           rates: Tuple[float, float,
                                        float, float,
                                        float, float],
                           iters: int) -> Tuple[np.ndarray, np.ndarray]:
    queue_lengths = np.zeros((sum(range(NumUsers)), iters))
    ql = np.zeros(iters)

    possible_matchings = []
    for edge in gen_network_graph(NumUsers):
        possible_matchings.append([edge])
    if H_num > 1:
        for matching in define_size_2_matchings(NumUsers):
            possible_matchings.append(matching)
    if H_num == 3:
        for matching in define_size_3_matchings(NumUsers):
            possible_matchings.append(matching)

    for x in range(iters):
        arrivals = generate_uniform_arivals(rate, NumUsers)
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

    return queue_lengths[:, iters - 1], ql


rate = 0.15
iters = 1000
rates = (rate, rate, rate, rate, rate, rate)
final_slice, ql = simulate_queue_lengths(4, 1, rates, iters)
final_slice2, ql2 = simulate_queue_lengths(4, 2, rates, iters)
final_slice3, ql3 = simulate_queue_lengths(4, 3, rates, iters)

plt.plot(range(iters), ql)
plt.plot(range(iters), ql2)
plt.plot(range(iters), ql3)
plt.show()
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
