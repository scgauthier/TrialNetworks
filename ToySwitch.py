import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import binom
from random import random
from typing import Tuple, List
from itertools import combinations, combinations_with_replacement
from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)


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
def get_size_h_schedules(h: int, NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations(range(pool_size), h))


# In case want to be able to schedule one VOQ more than once per slot
def get_size_h_flexible_schedules(h: int,
                                  NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations_with_replacement(range(pool_size), h))


def restrict_max_weight_search(marked: np.ndarray,
                               possible_schedules: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:
    infeasible = []
    for schedule in possible_schedules:
        for x in range(np.shape(marked)[0]):
            if (marked[x, 0] > 0) and (x in schedule):
                infeasible.append(schedule)
                break

    for schedule in infeasible:
        possible_schedules.remove(schedule)

    return possible_schedules


def select_max_weight_schedule(possible_schedules: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_schedule = np.zeros((np.shape(current_queue_lengths)[0], 1))
    max_weight = 0

    for schedule in possible_schedules:
        weight = 0
        array_schedule = np.zeros((np.shape(current_queue_lengths)[0], 1))
        for edge in schedule:
            weight += current_queue_lengths[edge]
            if current_queue_lengths[edge] > 0:
                array_schedule[edge, 0] = 1

        if weight > max_weight:
            max_weight = weight
            max_schedule = array_schedule
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_schedule = array_schedule
    return max_schedule


def model_probabilistic_link_gen(NumUsers: int, gen_prob: float,
                                 schedule: np.ndarray) -> list:
    new_schedule = np.copy(schedule)
    for x in range(int(bc(NumUsers, 2))):
        if schedule[x, 0] != 0:
            if random() > gen_prob:
                new_schedule[x, 0] = 0
    marked = schedule - new_schedule

    return new_schedule, marked


# failure_mechs: rq = return to queue, ss = stay scheduled
def simulate_queue_lengths(NumUsers: int, H_num: int,
                           max_subs: int,
                           pDist: str, prob_param: float,
                           gen_prob: float,
                           failure_mech: str,
                           iters: int) -> np.ndarray:
    queue_lengths = np.zeros((sum(range(NumUsers)), iters))
    ql = np.zeros(iters)

    possible_schedules = []
    for x in range(H_num + 1):
        possible_schedules += get_size_h_schedules(x, NumUsers)

    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        probs = [prob_param] * int(bc(NumUsers, 2))

    else:
        probs = [0] * int(bc(NumUsers, 2))

    num_failures = 0
    marked = np.zeros((int(bc(NumUsers, 2)), 1))
    for x in range(iters):
        arrivals = gen_arivals(max_subs, probs, NumUsers)
        schedule = np.zeros((int(NumUsers * (NumUsers - 1) / 2), 1))
        if (x > 0) and ((num_failures == 0) or
                        (failure_mech == 'rq')):
            schedule = select_max_weight_schedule(possible_schedules,
                                                  queue_lengths[:, x - 1])
        elif (x > 0) and (failure_mech == 'ss'):
            mod_possible = []
            for mod_h in range(H_num + 1 - num_failures):
                mod_possible += get_size_h_schedules(mod_h, NumUsers)

            mod_possible = restrict_max_weight_search(marked,
                                                      mod_possible,
                                                      queue_lengths[:, x - 1])

            schedule = select_max_weight_schedule(mod_possible,
                                                  queue_lengths[:, x - 1])

            schedule, marked = model_probabilistic_link_gen(NumUsers,
                                                            gen_prob,
                                                            (schedule
                                                             + marked))

        if (failure_mech == 'rq') or ((failure_mech == 'ss')
                                      and (num_failures == 0)):
            schedule, marked = model_probabilistic_link_gen(NumUsers,
                                                            gen_prob,
                                                            schedule)

        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:, 0]
                               - schedule[:, 0])

        num_failures = int(np.sum(marked))

        ql[x] = np.sum(queue_lengths[:, x], axis=0)

    return ql


# Want to define plotting function which will be able to show various scenarios
# with above and below threshold behaviour
# Fignames: LR = less restricted
def study_near_threshold(NumUsers: int, H_num: int, max_subs: int,
                         pDist: str, gen_prob: float, failure_mech: str,
                         iters: int, dist_fac: float) -> None:
    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        threshold = ((H_num * gen_prob / max_subs) * (1 / bc(NumUsers, 2))
                     // (1/10000)) / 10000  # Truncate at 4th place

    ds1 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold - (dist_fac * threshold),
                                 gen_prob, failure_mech, iters)

    ds2 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold, gen_prob,
                                 failure_mech, iters)

    ds3 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold + (dist_fac * threshold),
                                 gen_prob, failure_mech, iters)

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, m = {}, p = {}, T = {}'.format(
                 NumUsers, H_num, max_subs, gen_prob, threshold),
                 fontsize=28)
    ax1.plot(range(iters), ds1, color=cmap(0),
             label='T - {}'.format(dist_fac * threshold))
    ax2.plot(range(iters), ds2, color=cmap(inds[1]),
             label='T')
    ax3.plot(range(iters), ds3, color=cmap(inds[2]),
             label='T + {}'.format(dist_fac * threshold))

    ax3.legend(fontsize=22, framealpha=0.6, loc=2)

    ax2.legend(fontsize=22, framealpha=0.6, loc=2)

    ax1.legend(fontsize=22, framealpha=0.6, loc=2)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    p_whole = int(100 * gen_prob)
    figname = '../Figures/ProbGen_LR_{}_{}_{}_{}_{}'.format(NumUsers, H_num,
                                                            p_whole,
                                                            max_subs,
                                                            failure_mech)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


study_near_threshold(7, 3, 1, 'u', 0.75, 'rq', 100000, 0.05)
