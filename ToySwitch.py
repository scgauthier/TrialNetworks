import numpy as np
from scipy.stats import bernoulli
from typing import Tuple


# Assume lambda12=lambda21, 13=31, 14=41, 23=32, 24=42, 34=43
def generate_arivals(rates: Tuple[float, float,
                                  float, float,
                                  float, float]) -> np.ndarray:
    arrivals = np.zeros((6, 1))
    lambda12, lambda13, lambda14, lambda23, lambda24, lambda34 = rates

    arrivals[0, 0] = bernoulli.rvs(lambda12, size=1)[0]
    arrivals[1, 0] = bernoulli.rvs(lambda13, size=1)[0]
    arrivals[2, 0] = bernoulli.rvs(lambda14, size=1)[0]
    arrivals[3, 0] = bernoulli.rvs(lambda23, size=1)[0]
    arrivals[4, 0] = bernoulli.rvs(lambda24, size=1)[0]
    arrivals[5, 0] = bernoulli.rvs(lambda34, size=1)[0]

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


# ToDo: write function for determining the maximum weight matching

def simulate_queue_lengths(NumUsers: int, H_num: int,
                           rates: Tuple[float, float,
                                        float, float,
                                        float, float],
                           iters: int) -> np.ndarray:
    queue_lengths = np.zeros((sum(range(NumUsers)), iters))

    possible_matchings = gen_network_graph(NumUsers)
    if H_num > 1:
        possible_matchings.append(define_size_2_matchings(NumUsers))
    if H_num == 3:
        possible_matchings.append(define_size_3_matchings(NumUsers))

    for x in range(1, iters):
        arrivals = generate_arivals(rates)
        queue_lengths[:, x] = queue_lengths[:, x-1] + arrivals[:, 0]
    return queue_lengths


rates = (0.25, 0.25, 0.25, 0.25, 0.25, 0.25)
print(simulate_queue_lengths(4, 1, rates, 10))

# size = 4
# print('size 1: ', len(gen_network_graph(size)))
# print('size 2: ')
# print(len(define_size_2_matchings(size)), '\n\n',
#       define_size_2_matchings(size), '\n\n')
# print('size 3: ')
# print(len(define_size_3_matchings(size)), '\n\n',
#       define_size_3_matchings(size))
