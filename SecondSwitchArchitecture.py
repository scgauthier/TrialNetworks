import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.special import binom as bc
from random import sample, choice, shuffle, random
from math import floor
from typing import Tuple, List
from UniqueSizeHSchedules import gen_unique_schedules
from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)


# Function generates all unique pairs, each corresponding to a serparate VOQ
def gen_VOQ_graph(NumUsers: int) -> Tuple:
    Network_graph = []

    for requester in range(NumUsers - 1):
        for partner in range(requester + 1, NumUsers):
            Network_graph.append((requester, partner))

    return Network_graph


# Chooses uniformly at random N/2 unique pairings from the N nodes
# Used to assign which VOQs may have probabilistic arrivals in a time slot
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
            pairs.append((senders[x], reciever_pool[x]))
        else:
            pairs.append((reciever_pool[x], senders[x]))

    return pairs


# Generate arrivals according from time steps pairing. Arrivals are binomial,
# with a max number of requests submitted max_submissions and p_submit
def gen_arrivals(max_submissions: int,
                 probs: List[float],
                 NumUsers: int) -> np.ndarray:

    pairs = form_ordered_pairs(NumUsers)
    VOQs = gen_VOQ_graph(NumUsers)
    arrivals = np.zeros((len(VOQs), 1))

    for pair in pairs:
        ind = VOQs.index(pair)
        arrivals[ind, 0] = binom.rvs(max_submissions, probs[ind], size=1)[0]

    return arrivals


# Mathematical factorial
def fact(Number: int) -> int:
    factorial = 1
    for x in range(1, Number + 1):
        factorial *= x
    return factorial


# Number of matchings of size H
def calc_total_schedules(NumUsers: int, H_num: int) -> int:
    NumSchedules = (1 / fact(H_num))
    for m in range(H_num):
        NumSchedules *= bc(NumUsers - (2 * m), 2)
    return int(NumSchedules)


def select_max_weight_schedule(NumUsers: int, H_num: int,
                               current_queue_lengths: np.ndarray) -> Tuple:

    max_schedule = np.zeros((int(bc(NumUsers, 2)), 1))
    max_weight = 0

    schedules = []
    for x in range(1, H_num + 1):
        schedules += gen_unique_schedules(NumUsers, H_num)

    for schedule in schedules:
        weight = 0
        current = np.zeros((int(bc(NumUsers, 2)), 1))
        for VOQ_ind in schedule:
            weight += current_queue_lengths[VOQ_ind]
            if current_queue_lengths[VOQ_ind] > 0:
                current[VOQ_ind, 0] = 1

        if weight > max_weight:
            max_weight = weight
            max_schedule = current
        elif weight == max_weight:
            p = random()
            if p >= 0.5:
                max_schedule = current
    return max_schedule


def simulate_queue_lengths(NumUsers: int,
                           H_num: int,
                           max_submissions: int,
                           p_dist: str,
                           prob_param: float,
                           iters: int) -> Tuple[np.ndarray, np.ndarray]:

    queue_lengths = np.zeros((int(bc(NumUsers, 2)), iters))
    ql = np.zeros(iters)

    if ((p_dist == 'uniform') or (p_dist == 'Uniform') or (p_dist == 'u')
       or (p_dist == 'U')):
        probs = [prob_param] * int(bc(NumUsers, 2))

    else:
        probs = [0] * int(bc(NumUsers, 2))

    for x in range(iters):
        arrivals = gen_arrivals(max_submissions, probs, NumUsers)
        schedule = np.zeros((int(NumUsers * (NumUsers - 1) / 2), 1))
        if x > 0:
            schedule = select_max_weight_schedule(NumUsers,
                                                  H_num,
                                                  queue_lengths[:, x - 1])
        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:, 0]
                               - schedule[:, 0])

        ql[x] = np.sum(queue_lengths[:, x], axis=0)

    return ql


def calc_fraction_schedules(NumUsers: int, H_num: int) -> float:
    NumSchedules = calc_total_schedules(NumUsers, H_num)

    gamma = bc(NumUsers, 2) / (NumSchedules * H_num)

    return gamma


# Want to define plotting function which will be able to show various scenarios
# with above and below threshold behaviour
def study_near_threshold(NumUsers: int, H_num: int, max_subs: int,
                         pDist: str, iters: int, dist_fac: float) -> None:
    if ((pDist == 'uniform') or (pDist == 'Uniform') or (pDist == 'u')
       or (pDist == 'U')):
        # threshold = ((H_num / max_subs) * (2 / NumUsers)
        #              // (1/10000)) / 10000  # Truncate at 4th place
        threshold = 0.33

    ds1 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold - (dist_fac * threshold), iters)

    ds2 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold, iters)

    ds3 = simulate_queue_lengths(NumUsers, H_num, max_subs, pDist,
                                 threshold + (dist_fac * threshold), iters)

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, m = {},  T = {}'.format(NumUsers,
                                                          H_num,
                                                          max_subs,
                                                          threshold),
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

    figname = '../Figures/LR_{}_{}_{}'.format(NumUsers, H_num, max_subs)
    # plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()


# study_near_threshold(6, 3, 3, 'u', 5000, 0.06)
print(simulate_queue_lengths(4, 2, 1, 'u', 0.95, 100))
