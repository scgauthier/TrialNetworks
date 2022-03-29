import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import binom
from random import random, shuffle
from typing import Tuple, List
from itertools import combinations, combinations_with_replacement
from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)


# pDist options: 'uBin' = uniform binomial
#                'rBin' = random splitt of threshold amongst binomial NQs
def prep_dist(NumUsers: int, H_num: int,
              prob_param: float,
              pDist: str, max_scheduled: int) -> list:

    NQs = int(bc(NumUsers, 2))
    threshold = prob_param
    probs = [0] * NQs

    if pDist == 'uBin':
        probs = [prob_param / NQs] * NQs

    elif pDist == 'rBin':
        queues = list(range(NQs))
        shuffle(queues)

        max_val = max_scheduled * ((NQs / H_num) * prob_param) / NQs

        # ensure using full space
        while (round(sum(probs), 4) != round(threshold, 4)):
            count = 1
            prob_param = threshold
            for x in queues:
                if count < (NQs):
                    select = float(random())
                    incr = select * prob_param
                    while incr > max_val:
                        select = float(random())
                        incr = select * prob_param
                    probs[x] = incr
                    prob_param -= incr
                else:
                    # assign rest / prevent overflow
                    probs[x] = min(max_val, prob_param)
                count += 1
    return probs


# Allow arrivals to follow different distributions, specified by pDist.
# Specify max # of submissions per VOQ per round
# Probability of each request submission
# for queue i is p_i.
# pDist options: 'uBin' = uniform binomial
#                'rBin' = random splitt of threshold amongst binomial NQs
def gen_arrivals(max_submissions: int,
                 NumUsers: int,
                 pDist: str,
                 probs: list) -> np.ndarray:

    NQs = int(bc(NumUsers, 2))
    arrivals = np.zeros(NQs)

    if ((pDist == 'uBin') or (pDist == 'rBin')):

        for x in range(NQs):
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
        H_num -= num_failures

    max_length_inds = np.argpartition(QLs, -H_num)[-H_num:]
    # print(max_length_inds, 'max length inds')
    for x in max_length_inds:
        if QLs[x] > 0:  # don't sechedule empty queues
            max_lengths[x] = 1

    # print(max_lengths, 'schedule generated \n\n')
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
                      failure_mech: str,
                      max_sched_per_q: int) -> np.ndarray:

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

        # Check if max number of times scheduled reached
        if max_lengths[max_length_ind[0]] == max_sched_per_q:
            QLs[max_length_ind[0]] = 0

    return max_lengths


def weighted_strict_shortcut(H_num: int, NQs: int,
                             probs: list,
                             current_queue_lengths: np.ndarray,
                             marked: np.ndarray,
                             failure_mech: str,
                             service_times: np.ndarray,
                             time: int) -> np.ndarray:
    QLs = np.copy(current_queue_lengths)
    weights = np.zeros(NQs)

    # weight by time elapsed since service / requested rate
    for q in range(NQs):
        weights[q] = max(1, (time - service_times[q]) / probs[q])

    # print(QLs, 'start schedule queue lengths \n')
    max_lengths = np.zeros(NQs)
    if failure_mech == 'ss':
        for x in range(NQs):
            if marked[x] > 0:
                QLs[x] = 0  # Don't allow double scheduling
                max_lengths[x] = 1
        num_failures = int(np.sum(marked))
        H_num -= num_failures
        # print(marked, 'marked \n', QLs, 'new schedule queue lengths \n')
        # print(H_num, 'H \n')

    QLs = weights * QLs
    max_length_inds = np.argpartition(QLs, -H_num)[-H_num:]
    # print(max_length_inds, 'max length inds')
    for x in max_length_inds:
        if QLs[x] > 0:  # don't sechedule empty queues
            max_lengths[x] = 1

    # print(max_lengths, 'schedule generated \n\n')
    return max_lengths


def weighted_flex_shortcut(H_num: int, NQs: int,
                           probs: list,
                           current_queue_lengths: np.ndarray,
                           marked: np.ndarray,
                           failure_mech: str,
                           service_times: np.ndarray,
                           time: int,
                           max_sched_per_q: int) -> np.ndarray:

    QLs = np.copy(current_queue_lengths)
    max_lengths = np.zeros(NQs)
    weights = np.zeros(NQs)

    # weight by time elapsed since service / requested rate
    for q in range(NQs):
        weights[q] = max(1, (time - service_times[q]) / probs[q])

    if failure_mech == 'ss':
        QLs = QLs - marked
        num_failures = int(np.sum(marked))
        max_lengths = marked
        H_num -= num_failures

    QLs = weights * QLs
    for x in range(H_num):
        max_length_ind = np.argpartition(QLs, -1)[-1:]
        # don't sechedule empty queues
        if QLs[max_length_ind[0]] > 0:
            max_lengths[max_length_ind[0]] += 1.0
        # Cost function -1 to prevent scheduling same q H times
        QLs[max_length_ind[0]] -= 1

        # Check if max number of times scheduled reached
        if max_lengths[max_length_ind[0]] == max_sched_per_q:
            QLs[max_length_ind[0]] = 0

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


############################################################
# Main simulation ##########################################

# failure_mechs: rq = return to queue, ss = stay scheduled
# sched_type: base or weighted
def simulate_queue_lengths(NumUsers: int, H_num: int,
                           max_subs: int,
                           pDist: str, prob_param: float,
                           gen_prob: float,
                           failure_mech: str,
                           sched_type: str,
                           max_sched_per_q: int,
                           iters: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, list, list,
                                                list]:

    NQs = int(bc(NumUsers, 2))
    queue_lengths = np.zeros((NQs, iters))
    submission_times = np.zeros((NQs, iters))
    waiting_times_perQ = np.zeros((2, NQs))
    observed_times_perQ = np.zeros((3, NQs))
    waiting_dist_max = []
    waiting_dist_min = []
    avrg_rates = np.zeros(NQs)
    service_times = np.zeros(NQs)
    schedule = np.zeros(NQs)
    marked = np.zeros(NQs)

    probs = prep_dist(NumUsers, H_num, prob_param, pDist, max_sched_per_q)
    ratemax, ratemin = probs.index(max(probs)), probs.index(min(probs))
    print('\n\n Requested', probs, 'Sum: ', sum(probs))

    for x in range(iters):
        # Get submitted requests
        arrivals = gen_arrivals(max_subs, NumUsers, pDist, probs)
        # Update submission times
        for y in np.nonzero(arrivals):
            submission_times[y, x] = arrivals[y]

        # for current schedule, do link gen
        schedule, marked = model_probabilistic_link_gen(NumUsers,
                                                        gen_prob,
                                                        schedule)

        # Update service times:
        for y in np.nonzero(schedule)[0]:
            for z in range(int(schedule[y])):
                try:
                    subs = np.nonzero(submission_times[y, 0:x])[0]
                    # Update waiting times for served request
                    # -1 since can't be served until next slot
                    waiting_times_perQ[0, y] += max(x - subs[0] - 1,
                                                    0)
                    # Clear served requests from tracking
                    submission_times[y, subs[0]] -= 1
                    # Update number requests served
                    waiting_times_perQ[1, y] += 1
                except IndexError:
                    continue
                # Sum service intervals
                service_int = x - observed_times_perQ[2, y]
                observed_times_perQ[0, y] += service_int
                observed_times_perQ[1, y] += 1
                observed_times_perQ[2, y] = x  # update service time

                if y == ratemax:
                    waiting_dist_max.append(service_int)
                elif y == ratemin:
                    waiting_dist_min.append(service_int)

            service_times[y] = x

        # Update queue lengths at x based on lengths at x-1, schedule from x-1,
        # successful link gen at x, and arrivals at x
        # Essentially lengths by end of x
        queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:]
                               - schedule[:])

        # replace with schedule types normal and flexible and
        # specification of max scheduled per q

        if sched_type == 'base':
            schedule = flexible_shortcut(H_num, NQs,
                                         queue_lengths[:, x],
                                         marked,
                                         failure_mech,
                                         max_sched_per_q)
        elif sched_type == 'weighted':
            schedule = weighted_flex_shortcut(H_num, NQs,
                                              probs,
                                              queue_lengths[:, x],
                                              marked,
                                              failure_mech,
                                              service_times,
                                              x,
                                              max_sched_per_q)
        else:
            print('Invalid scheduling type specified')
            return

    for x in range(NQs):
        if waiting_times_perQ[1, x] > 0:
            waiting_times_perQ[0, x] = (waiting_times_perQ[0, x]
                                        / waiting_times_perQ[1, x])
        avrg_rates[x] = waiting_times_perQ[1, x] / iters

        if observed_times_perQ[1, x] > 0:
            observed_times_perQ[0, x] = (observed_times_perQ[0, x]
                                         / observed_times_perQ[1, x])

    return (queue_lengths, observed_times_perQ[0, :], avrg_rates,
            (probs * max_subs), waiting_dist_max, waiting_dist_min)


##########################################################################
# Begin Analysis ######################################################

def plot_queue_stability(q1: np.ndarray,
                         q2: np.ndarray,
                         q3: np.ndarray,
                         NumUsers: int,
                         H_num: int,
                         max_subs: int,
                         gen_prob: float,
                         threshold: float,
                         dist_fac: float,
                         iters: int,
                         figname: str) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, m = {}, p = {}, T = {}'.format(
                 NumUsers, H_num, max_subs, gen_prob, threshold),
                 fontsize=28)
    ax1.plot(range(iters), q1, color=cmap(0),
             label='T - {}'.format(dist_fac * threshold))
    ax2.plot(range(iters), q2, color=cmap(inds[1]),
             label='T')
    ax3.plot(range(iters), q3, color=cmap(inds[2]),
             label='T + {}'.format(dist_fac * threshold))

    ax3.legend(fontsize=22, framealpha=0.6, loc=2)

    ax2.legend(fontsize=22, framealpha=0.6, loc=2)

    ax1.legend(fontsize=22, framealpha=0.6, loc=2)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_waiting_time_dists(max_dist: list,
                            min_dist: list,
                            rates: list,
                            NumUsers: int,
                            H_num: int,
                            max_subs: int,
                            figname: str) -> None:

    cmap = plt.cm.get_cmap('plasma')
    fig, (ax1, ax2) = plt.subplots(2, sharex=False, figsize=(10, 8))

    ax1.plot(range(len(max_dist)), max_dist, color=cmap(0),
             label='Max Rate Request')
    ax1.plot(range(len(max_dist)), len(max_dist)*[1/(max(rates))],
             '--', color=cmap(0.9))

    ax2.plot(range(len(min_dist)), min_dist, color=cmap(0.5),
             label='Min Rate Request')
    ax2.plot(range(len(min_dist)), len(min_dist)*[1/min(rates)],
             '--', color=cmap(0.9))

    ax2.legend(fontsize=22, framealpha=0.6, loc=2)

    ax1.legend(fontsize=22, framealpha=0.6, loc=2)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_individual_queues(q1: np.ndarray,
                           q2: np.ndarray,
                           q3: np.ndarray,
                           iters: int,
                           NumUsers: int, H_num: int, max_subs: int,
                           wt: np.ndarray, rt: np.ndarray,
                           rr: list, dist_fac: float,
                           threshold: float,
                           sched_type: str) -> None:

    NQs = int(bc(NumUsers, 2))
    cmap = plt.cm.get_cmap('plasma')
    plt.figure(figsize=(10, 8))

    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q1[x, :], color=cmap(inds[x]),
                 label='WT={}, Rate={}, RR={}'.format(round(wt[0, x], 1),
                                                      round(rt[0, x], 3),
                                                      round(rr[0, x], 3)))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=1)
    plt.title('T - {}'.format((((dist_fac * threshold) // (1/1000)) / 1000)),
              fontsize=28)
    figname = '../Figures/PairRequest/IQs_LR_{}_{}_{}_{}_BelowT'.format(
            NumUsers, H_num, max_subs, sched_type)
    plt.savefig(figname, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q2[x, :], color=cmap(inds[x]),
                 label='WT={}, Rate={}, RR={}'.format(round(wt[1, x], 1),
                                                      round(rt[1, x], 3),
                                                      round(rr[1, x], 3)))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=1)
    plt.title('T', fontsize=28)
    figname = '../Figures/PairRequest/IQs_LR_{}_{}_{}_{}_AtT'.format(
            NumUsers, H_num, max_subs, sched_type)
    plt.savefig(figname, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q3[x, :], color=cmap(inds[x]),
                 label='WT={}, Rate={}, RR={}'.format(round(wt[2, x], 1),
                                                      round(rt[2, x], 3),
                                                      round(rr[2, x], 3)))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=2)
    plt.title('T + {}'.format((((dist_fac * threshold) // (1/1000)) / 1000)),
              fontsize=28)
    figname = '../Figures/PairRequest/IQs_LR_{}_{}_{}_{}_AboveT'.format(
            NumUsers, H_num, max_subs, sched_type)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


# Want to define plotting function which will be able to show various scenarios
# with above and below threshold behaviour
# Fignames: LR = less restricted
def study_near_threshold(NumUsers: int, H_num: int, max_subs: int,
                         pDist: str, gen_prob: float, failure_mech: str,
                         sched_type: str, max_sched_per_q: int,
                         iters: int,
                         dist_fac: float) -> None:

    threshold = ((H_num * gen_prob / max_subs)
                 // (1/10000)) / 10000  # Truncate at 4th place

    (q1, wt1, rt1, rr1,
     waitMax1, waitMin1) = simulate_queue_lengths(NumUsers, H_num, max_subs,
                                                  pDist,
                                                  (1 - dist_fac) * (threshold),
                                                  gen_prob, failure_mech,
                                                  sched_type,
                                                  max_sched_per_q, iters)
    (q2, wt2, rt2, rr2,
     waitMax2, waitMin2) = simulate_queue_lengths(NumUsers, H_num, max_subs,
                                                  pDist, threshold, gen_prob,
                                                  failure_mech,
                                                  sched_type,
                                                  max_sched_per_q, iters)
    (q3, wt3, rt3, rr3,
     waitMax3, waitMin3) = simulate_queue_lengths(NumUsers, H_num, max_subs,
                                                  pDist,
                                                  (1 + dist_fac) * threshold,
                                                  gen_prob, failure_mech,
                                                  sched_type,
                                                  max_sched_per_q, iters)

    pltQStab = True
    if pltQStab:

        ql1, ql2, ql3 = np.zeros(iters), np.zeros(iters), np.zeros(iters)
        for x in range(iters):
            # Store total queue lengths
            ql1[x] = np.sum(q1[:, x], axis=0)
            ql2[x] = np.sum(q2[:, x], axis=0)
            ql3[x] = np.sum(q3[:, x], axis=0)

        p_whole = int(100 * gen_prob)
        figname = '../Figures/PairRequest/QStab_LR_{}_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, p_whole, max_subs, pDist, sched_type)

        plot_queue_stability(ql1, ql2, ql3, NumUsers, H_num, max_subs,
                             gen_prob, threshold, dist_fac, iters, figname)

    pltWTD = True
    if pltWTD:
        figname = '../Figures/PairRequest/LR_WTD_{}_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, max_subs, sched_type,
                round(1/max(rates)), round(1/min(rates)))
        plot_waiting_time_dists(waitMax1, waitMin1, rt1,
                                NumUsers, H_num, max_subs, figname)

    pltIdvQs = True
    if pltIdvQs:

        wt = np.array([wt1, wt2, wt3])
        rt = np.array([rt1, rt2, rt3])
        rr = np.array([rr1, rr2, rr3])

        plot_individual_queues(q1, q2, q3, iters, NumUsers,
                               H_num, max_subs,
                               wt, rt, rr,
                               dist_fac, threshold, sched_type)

    print(wt1, 1/rt1, rt1, sum(rt1), '\n\n', wt2, 1/rt2, rt2, sum(rt2),
          '\n\n', wt3, 1/rt3, rt3, sum(rt3))


# study_near_threshold(4, 2, 1, 'rBin', 0.75, 'rq', 'strict', 1, 100000, 0.05)
# print(simulate_queue_lengths(4, 2, 1, 'rBin', 6/4, 0.75, 'rq', 'strict', 10))
