import numpy as np
from math import floor, ceil
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import binom
from random import random, shuffle
from typing import Tuple
# from itertools import combinations, combinations_with_replacement
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

        max_val = max_scheduled * (prob_param / H_num)

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
def gen_arrivals(NumUsers: int,
                 probs: list,
                 params: dict) -> np.ndarray:

    NQs = ceil(int(bc(NumUsers, 2)) * params['sessionSamples'])
    arrivals = np.zeros(NQs)

    for x in range(NQs):
        if probs[x] < 1:
            arrivals[x] = binom.rvs(1, probs[x], size=1)[0]
        else:  # deal with rates > 1
            int_part = floor(probs[x])
            prob = probs[x] - floor(probs[x])
            arrivals[x] = (int_part
                           + binom.rvs(1, prob, size=1)[0])

    return arrivals


# For the less restricted switch with flexible scheduling, the MaxWeight
# scheduling procedure simply reduces to first choosing the longest queue
# and scheduling it, reducing the queue length in the calculation by 1,
# and repeating H-1 times.
# This method performs much better for large NQs and H>2
def flexible_schedule(H_num: int,
                      NQs: int,
                      current_queue_lengths: np.ndarray,
                      max_sched_per_q: int) -> np.ndarray:

    QLs = np.copy(current_queue_lengths)
    max_lengths = np.zeros(NQs)

    # for making tie breaking random in argpartition
    # at paired up level, could insert np.non-zero here on QLs
    paired_up = []
    for x in range(NQs):
        paired_up.append((QLs[x], x))
    shuffle(paired_up)
    for x in range(NQs):
        QLs[x] = paired_up[x][0]

    for x in range(H_num):
        max_length_ind = np.argpartition(QLs, -1)[-1:]
        actual_ind = paired_up[max_length_ind[0]][1]
        # don't sechedule empty queues
        if QLs[max_length_ind[0]] > 0:
            max_lengths[actual_ind] += 1.0
        # Cost function -1 to prevent scheduling same q H times
        QLs[max_length_ind[0]] -= 1

        # Check if max number of times scheduled reached
        if max_lengths[actual_ind] == max_sched_per_q:
            QLs[max_length_ind[0]] = 0

    return max_lengths


def weighted_flex_schedule(H_num: int, NQs: int,
                           probs: list,
                           current_queue_lengths: np.ndarray,
                           service_times: np.ndarray,
                           time: int,
                           max_sched_per_q: int) -> np.ndarray:

    QLs = np.copy(current_queue_lengths)
    max_lengths = np.zeros(NQs)
    weights = np.zeros(NQs)

    # weight by time elapsed since service / requested rate
    for q in range(NQs):
        weights[q] = max(1, (time - service_times[q]) / probs[q])

    # apply weights
    QLs = weights * QLs

    # for making tie breaking random in argpartition
    paired_up = []
    for x in range(NQs):
        paired_up.append((QLs[x], x))
    shuffle(paired_up)
    for x in range(NQs):
        QLs[x] = paired_up[x][0]

    for x in range(H_num):
        max_length_ind = np.argpartition(QLs, -1)[-1:]
        actual_ind = paired_up[max_length_ind[0]][1]
        # don't sechedule empty queues
        if QLs[max_length_ind[0]] > 0:
            max_lengths[actual_ind] += 1.0
        # Cost function -1 to prevent scheduling same q H times
        QLs[max_length_ind[0]] -= 1

        # Check if max number of times scheduled reached
        if max_lengths[actual_ind] == max_sched_per_q:
            QLs[max_length_ind[0]] = 0

    return max_lengths


def model_probabilistic_link_gen(NumUsers: int,
                                 gen_prob: float,
                                 params: dict,
                                 schedule: np.ndarray) -> list:
    new_schedule = np.copy(schedule)
    for x in range(ceil(int(bc(NumUsers, 2)) * params['sessionSamples'])):
        if schedule[x] != 0:
            for y in range(int(schedule[x])):  # accomodate flexible scheduling
                if random() > gen_prob:
                    new_schedule[x] -= 1

    return new_schedule


############################################################
# Main simulation ##########################################

# sched_type: base or weighted
def simulate_queue_lengths(NumUsers: int, H_num: int,
                           pDist: str, prob_param: float,
                           gen_prob: float,
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

    probs = prep_dist(NumUsers, H_num, prob_param, pDist, max_sched_per_q)
    ratemax, ratemin = probs.index(max(probs)), probs.index(min(probs))
    print('\n\n Requested', probs, 'Sum: ', sum(probs))

    for x in range(iters):
        # Get submitted requests
        arrivals = gen_arrivals(NumUsers, probs)
        # Update submission times
        for y in np.nonzero(arrivals):
            submission_times[y, x] = arrivals[y]

        # for current schedule, do link gen
        schedule = model_probabilistic_link_gen(NumUsers,
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
            schedule = flexible_schedule(H_num, NQs,
                                         queue_lengths[:, x],
                                         max_sched_per_q)
        elif sched_type == 'weighted':
            schedule = weighted_flex_schedule(H_num, NQs,
                                              probs,
                                              queue_lengths[:, x],
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
            probs, waiting_dist_max, waiting_dist_min)


##########################################################################
# Begin Analysis ######################################################

def plot_queue_stability(q1: np.ndarray,
                         q2: np.ndarray,
                         q3: np.ndarray,
                         NumUsers: int,
                         H_num: int,
                         gen_prob: float,
                         threshold: float,
                         dist_fac: float,
                         iters: int,
                         figname: str) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, p = {}, T = {}'.format(
                 NumUsers, H_num, gen_prob, threshold),
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
                            rates: list,
                            NumUsers: int,
                            H_num: int,
                            figname: str) -> None:

    # Avrg over samples
    NS = 10
    scaling = 1.01
    # min_Av, max_Av = [1/min(rates)] * (NS - 1), [1/max(rates)] * (NS - 1)
    max_Av = [1/max(rates)] * (NS - 1)
    for x in range(len(max_dist) - (NS - 1)):
        sum = 0
        for y in range(NS):
            sum += max_dist[x - y]
        max_Av.append(sum / NS)

    # for x in range(len(min_dist) - (NS - 1)):
    #     sum = 0
    #     for y in range(NS):
    #         sum += min_dist[x - y]
    #     min_Av.append(sum / NS)

    cmap = plt.cm.get_cmap('plasma')
    plt.figure(figsize=(10, 8))
    # fig, (ax1, ax2) = plt.subplots(2, sharex=False, figsize=(10, 8))

    plt.plot(range(len(max_dist)), max_dist, color=cmap(0),
             label='Max Rate Request')
    plt.plot(range(len(max_dist)), max_Av, color=cmap(0.75),
             label='{} Point Av'.format(NS))
    plt.plot(range(len(max_dist)), len(max_dist)*[scaling/(max(rates))],
             '--', color=cmap(0.9))

    counts_over = 0
    for x in max_Av:
        if x > (scaling / max(rates)):
            counts_over += 1
    print("Counts over = ", counts_over)

    # ax1.plot(range(len(max_dist)),
    #          len(max_dist)*[1.5 / max(rates)],
    #          '--', color=cmap(0.95))

    # ax2.plot(range(len(min_dist)), min_dist, color=cmap(0.5),
    #          label='Min Rate Request')
    # ax2.plot(range(len(min_dist)), min_Av, color=cmap(0.25),
    #          label='{} Point Av'.format(NS))
    # ax2.plot(range(len(min_dist)), len(min_dist)*[1/min(rates)],
    #          '--', color=cmap(0.9))
    # ax2.plot(range(len(min_dist)),
    #          len(min_dist)*[2.5 / min(rates)],
    #          '--', color=cmap(0.95))
    #
    # ax2.legend(fontsize=22, framealpha=0.6, loc=2)
    #
    plt.legend(fontsize=22, framealpha=0.6, loc=2)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_individual_queues(q1: np.ndarray,
                           q2: np.ndarray,
                           q3: np.ndarray,
                           iters: int,
                           NumUsers: int, H_num: int,
                           dist_fac: float,
                           threshold: float) -> None:

    NQs = int(bc(NumUsers, 2))
    cmap = plt.cm.get_cmap('plasma')
    plt.figure(figsize=(10, 8))

    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q1[x, :], color=cmap(inds[x]))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=1)
    plt.title('T - {}'.format((((dist_fac * threshold) // (1/1000)) / 1000)),
              fontsize=28)
    figname = '../Figures/AlgAdjust/IQs_LR_{}_{}_BelowT'.format(
            NumUsers, H_num)
    plt.savefig(figname, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q2[x, :], color=cmap(inds[x]))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=1)
    plt.title('T', fontsize=28)
    figname = '../Figures/AlgAdjust/IQs_LR_{}_{}_AtT'.format(
            NumUsers, H_num)
    plt.savefig(figname, dpi=300, bbox_inches='tight')

    plt.figure(figsize=(10, 8))
    inds = np.linspace(0, 0.85, NQs)
    for x in range(NQs):
        plt.plot(range(iters), q3[x, :], color=cmap(inds[x]))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20, framealpha=0.6, loc=2)
    plt.title('T + {}'.format((((dist_fac * threshold) // (1/1000)) / 1000)),
              fontsize=28)
    figname = '../Figures/AlgAdjust/IQs_LR_{}_{}_AboveT'.format(
            NumUsers, H_num)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


# Want to define plotting function which will be able to show various scenarios
# with above and below threshold behaviour
# Fignames: LR = less restricted
def study_near_threshold(NumUsers: int, H_num: int,
                         pDist: str, gen_prob: float,
                         sched_type: str, max_sched_per_q: int,
                         iters: int,
                         dist_fac: float) -> None:

    threshold = ((H_num * gen_prob)
                 // (1/10000)) / 10000  # Truncate at 4th place

    (q1, wt1, rt1, rr1,
     waitMax1, waitMin1) = simulate_queue_lengths(NumUsers, H_num,
                                                  pDist,
                                                  (1 - dist_fac) * (threshold),
                                                  gen_prob, sched_type,
                                                  max_sched_per_q, iters)
    (q2, wt2, rt2, rr2,
     waitMax2, waitMin2) = simulate_queue_lengths(NumUsers, H_num,
                                                  pDist, threshold, gen_prob,
                                                  sched_type,
                                                  max_sched_per_q, iters)
    (q3, wt3, rt3, rr3,
     waitMax3, waitMin3) = simulate_queue_lengths(NumUsers, H_num,
                                                  pDist,
                                                  (1 + dist_fac) * threshold,
                                                  gen_prob, sched_type,
                                                  max_sched_per_q, iters)

    pltQStab = False
    if pltQStab:

        ql1, ql2, ql3 = np.zeros(iters), np.zeros(iters), np.zeros(iters)
        for x in range(iters):
            # Store total queue lengths
            ql1[x] = np.sum(q1[:, x], axis=0)
            ql2[x] = np.sum(q2[:, x], axis=0)
            ql3[x] = np.sum(q3[:, x], axis=0)

        p_whole = int(100 * gen_prob)
        figname = '../Figures/PairRequest/QStab_LR_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, p_whole, pDist, sched_type)

        plot_queue_stability(ql1, ql2, ql3, NumUsers, H_num,
                             gen_prob, threshold, dist_fac, iters, figname)

    pltWTD = True
    if pltWTD:
        figname = '../Figures/PairRequest/BT_WTD_{}_{}_{}'.format(
                NumUsers, H_num, sched_type)
        plot_waiting_time_dists(waitMax1, rr1,
                                NumUsers, H_num, figname)
        figname = '../Figures/PairRequest/AT_WTD_{}_{}_{}'.format(
                NumUsers, H_num, sched_type)
        plot_waiting_time_dists(waitMax3, rr3,
                                NumUsers, H_num, figname)

    pltIdvQs = False
    if pltIdvQs:

        wt = np.array([wt1, wt2, wt3])
        rt = np.array([rt1, rt2, rt3])
        rr = np.array([rr1, rr2, rr3])

        plot_individual_queues(q1, q2, q3, iters, NumUsers,
                               H_num,
                               wt, rt, rr,
                               dist_fac, threshold, sched_type)

    # print(wt1, 1/rt1, rt1, sum(rt1), '\n\n', wt2, 1/rt2, rt2, sum(rt2),
    #       '\n\n', wt3, 1/rt3, rt3, sum(rt3))


# study_near_threshold(4, 2, 'uBin', 0.75, 'base', 2, 10000, 0.05)
