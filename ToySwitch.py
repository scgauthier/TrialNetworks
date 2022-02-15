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
    arrivals = np.zeros(cutoff)

    for x in range(cutoff):
        arrivals[x] = binom.rvs(max_submissions, probs[x], size=1)[0]

    return arrivals


# Here network graph just corresponds to all ordered pairs of nodes, i.e.
# the distinct VOQs
def gen_network_graph(NumUsers: int) -> Tuple:
    Network_graph = []

    for requester in range(NumUsers-1):
        for partner in range(requester, NumUsers-1):
            Network_graph.append((requester, partner))

    return Network_graph


# All distinct ways of choosing h different VOQs
# Used to generate pool of possible schedules
def get_size_h_schedules(h: int, NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations(range(pool_size), h))


# In case want to be able to schedule one VOQ more than once per slot
# Used to generate pool of possible schedules
def get_size_h_flexible_schedules(h: int,
                                  NumUsers: int) -> list:
    pool_size = int(bc(NumUsers, 2))
    return list(combinations_with_replacement(range(pool_size), h))


# Used when failure mechanism is 'stay scheduled' (and scheduling not flexible)
# Removes the queues staying scheduled from schedule search space
def restrict_max_weight_search(marked: np.ndarray,
                               possible_schedules: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:
    infeasible = []
    for schedule in possible_schedules:
        for x in range(np.shape(marked)[0]):
            if (marked[x] > 0) and (x in schedule):
                infeasible.append(schedule)
                break

    for schedule in infeasible:
        possible_schedules.remove(schedule)

    return possible_schedules


def select_max_weight_schedule(possible_schedules: Tuple,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_schedule = np.zeros(np.shape(current_queue_lengths)[0])
    max_weight = 0

    for schedule in possible_schedules:
        weight = 0
        array_schedule = np.zeros(np.shape(current_queue_lengths)[0])
        current_lengths = np.copy(current_queue_lengths)
        for edge in schedule:
            weight += current_lengths[edge]
            if current_lengths[edge] > 0:
                array_schedule[edge] += 1
            current_lengths[edge] -= 1

        if weight > max_weight:
            max_weight = weight
            max_schedule = array_schedule
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_schedule = array_schedule
    return max_schedule


# For the less restricted switch with inflexible scheduling, the MaxWeight
# scheduling procedure simply reduces to choosing the H longest queues for
# scheduling at each time step
# This method performs much better for large NQs and H>2
def shortcut_max_weight_schedule(H_num: int,
                                 NQs: int,
                                 current_queue_lengths: np.ndarray,
                                 marked: np.ndarray,
                                 failure_mech: str) -> np.ndarray:

    QLs = np.copy(current_queue_lengths)
    max_lengths = np.zeros(NQs)
    if failure_mech == 'ss':
        for x in range(NQs):
            if marked[x] > 0:
                QLs[x] = 0  # Don't allow double scheduling
                max_lengths[x] = 1
        num_failures = int(np.sum(marked))
        max_lengths = marked
        H_num -= num_failures

    max_length_inds = np.argpartition(QLs, -H_num)[-H_num:]
    for x in max_length_inds:
        if QLs[x] > 0:  # don't sechedule empty queues
            max_lengths[x] = 1

    return max_lengths


# For the less restricted switch with flexible scheduling, the MaxWeight
# scheduling procedure simply reduces to first choosing the longest queue
# and scheduling it, reducing the queue length in the calculation by 1,
# and repeating H-1 times.
# This method performs much better for large NQs and H>2
def flexible_shortcut(H_num: int,
                      NQs: int,
                      current_queue_lengths: np.ndarray,
                      marked: np.ndarray,
                      failure_mech: str) -> np.ndarray:

    QLs = np.copy(current_queue_lengths)
    max_lengths = np.zeros(NQs)
    if failure_mech == 'ss':
        QLs = QLs - marked
        num_failures = int(np.sum(marked))
        max_lengths = marked
        H_num -= num_failures

    for x in range(H_num):
        max_length_ind = np.argpartition(QLs, -1)[-1:]
        # don't sechedule empty queues
        if QLs[max_length_ind[0]] > 0:
            max_lengths[max_length_ind[0]] += 1.0
        # Cost function -1 to prevent scheduling same q H times
        QLs[max_length_ind[0]] -= 1

    return max_lengths


def model_probabilistic_link_gen(NumUsers: int, gen_prob: float,
                                 schedule: np.ndarray) -> list:
    new_schedule = np.copy(schedule)
    for x in range(int(bc(NumUsers, 2))):
        if schedule[x] != 0:
            for y in range(int(schedule[x])):  # accomodate flexible scheduling
                if random() > gen_prob:
                    new_schedule[x] -= 1
    marked = schedule - new_schedule  # marks failed generations

    return new_schedule, marked


# failure_mechs: rq = return to queue, ss = stay scheduled
def simulate_queue_lengths(NumUsers: int, H_num: int,
                           max_subs: int,
                           pDist: str, prob_param: float,
                           gen_prob: float,
                           failure_mech: str,
                           sched_type: str,
                           iters: int) -> np.ndarray:

    NQs = int(bc(NumUsers, 2))
    queue_lengths = np.zeros((NQs, iters))
    ql = np.zeros(iters)
    submission_times = np.zeros((NQs, iters))
    waiting_times_perQ = np.zeros((2, NQs))
    cumulative_rates = np.zeros(NQs)
    schedule = np.zeros(NQs)
    marked = np.zeros(NQs)

    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        probs = [prob_param] * NQs

    else:
        probs = [0] * NQs

    for x in range(iters):
        # Get submitted requests
        arrivals = gen_arivals(max_subs, probs, NumUsers)
        # Update submission times
        for y in np.nonzero(arrivals):
            submission_times[y, x] = 1

        # for current schedule, do link gen
        schedule, marked = model_probabilistic_link_gen(NumUsers,
                                                        gen_prob,
                                                        schedule)

        # Update service times:
        for y in np.nonzero(schedule)[0]:
            try:
                subs = np.nonzero(submission_times[y, 0:x])[0]
                # Update waiting times for served request
                waiting_times_perQ[0, y] += x - subs[0]
                # Clear served requests from tracking
                submission_times[y, subs[0]] = 0
                # Update number requests served
                waiting_times_perQ[1, y] += 1
            except IndexError:
                continue

        # Update queue lengths at x based on lengths at x-1, schedule from x-1,
        # successful link gen at x, and arrivals at x
        # Essentially lengths by end of x
        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:]
                               - schedule[:])

        # Store total queue lengths
        ql[x] = np.sum(queue_lengths[:, x], axis=0)

        if sched_type == 'strict':
            schedule = shortcut_max_weight_schedule(H_num, NQs,
                                                    queue_lengths[:, x],
                                                    marked,
                                                    failure_mech)
        elif sched_type == 'flexible':
            schedule = flexible_shortcut(H_num, NQs,
                                         queue_lengths[:, x],
                                         marked,
                                         failure_mech)
        else:
            print('Invalid scheduling type specified')
            return

    for x in range(NQs):
        if waiting_times_perQ[1, x] > 0:
            waiting_times_perQ[0, x] = (waiting_times_perQ[0, x]
                                        / waiting_times_perQ[1, x])
        cumulative_rates[x] = waiting_times_perQ[1, x] / iters

    return ql, waiting_times_perQ[0, :], cumulative_rates


# Want to define plotting function which will be able to show various scenarios
# with above and below threshold behaviour
# Fignames: LR = less restricted
def study_near_threshold(NumUsers: int, H_num: int, max_subs: int,
                         pDist: str, gen_prob: float, failure_mech: str,
                         sched_type: str, iters: int,
                         dist_fac: float) -> None:
    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        threshold = ((H_num * gen_prob / max_subs) * (1 / bc(NumUsers, 2))
                     // (1/10000)) / 10000  # Truncate at 4th place

    ds1, wt1, rt1 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                           threshold - (dist_fac * threshold),
                                           gen_prob, failure_mech,
                                           sched_type, iters)
    ds2, wt2, rt2 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                           threshold, gen_prob,
                                           failure_mech,
                                           sched_type,
                                           iters)
    ds3, wt3, rt3 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                           threshold + (dist_fac * threshold),
                                           gen_prob, failure_mech,
                                           sched_type, iters)

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
    figname = '../Figures/PairRequest/ProbGen_LR_{}_{}_{}_{}_{}_{}'.format(
            NumUsers, H_num, p_whole, max_subs, failure_mech, sched_type)

    plt.savefig(figname, dpi=300, bbox_inches='tight')
    print(wt1, rt1, '\n\n', wt2, rt2, '\n\n', wt3, rt3)


study_near_threshold(4, 2, 1, 'u', 0.75, 'rq', 'flexible', 100000, 0.05)
# print(simulate_queue_lengths(4, 2, 1, 'u', 0.25, 0.75, 'ss', 'strict', 100))
