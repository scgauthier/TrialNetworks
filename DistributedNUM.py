import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from typing import Tuple, List

from ToySwitch import gen_arrivals, model_probabilistic_link_gen
from ToySwitch import flexible_schedule
from ToySwitch import plot_queue_stability, plot_individual_queues
from ToySwitch import plot_waiting_time_dists

from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)

# Base simulation off of sim in ToySwitch, but allow rates to vary.
# Also try to allow using either scheduling mode from beginning

# What should simulation do?
# Should adjust rates according to pricing algorithm
# What information is needed?
# p_c: requires each session to be provided with total queue lengths of
# each previous time step, as well as sum of rates from previous time step
# p_u: requires each user to sum over all queue lengths belonging to their
# sessions, as well as to sum their requested rates over their sessions.


# First need to implement session id to user matching algorithm
# Currently, sessions index from 1, users from 0
def users_to_session_id(NumUsers: int, user_u: int,
                        user_v: int) -> int:

    session_id = user_v - user_u
    for x in range(user_u):
        session_id += NumUsers - x - 1

    return session_id


def session_id_to_users(NumUsers: int, session_id: int) -> list[int]:

    # start with block 1
    x = 0
    remaining = session_id
    subtracted = 0

    while remaining > 0:

        x += 1
        # block one is sessions 1 to N-1
        if x == 1:
            remaining -= NumUsers - 1

        else:
            remaining -= NumUsers - x
            subtracted += NumUsers - x + 1

    y = session_id + x - subtracted

    return [x - 1, y - 1]


# takes as input the total number of sessions
# Outputs a list of lists, ordered by user number, of S(u) for each u.
def partition_sessions_by_user(NumUsers: int, NQs: int) -> np.ndarray:

    userSessions = np.zeros((NumUsers, NumUsers - 1))
    for x in range(NQs):
        min_entries = np.argmin(userSessions, axis=1)
        user_pair = session_id_to_users(NumUsers, x + 1)
        for u in user_pair:
            userSessions[u, min_entries[u]] = x + 1

    return userSessions


def update_prices(NumUsers: int, userSessions: np.ndarray,
                  lambda_Switch: float, user_max_rates: list,
                  lastT_queue_lengths: np.ndarray,
                  lastT_rates: np.ndarray) -> list[int]:

    sum_session_rates = np.sum(lastT_rates)
    sum_ql = np.sum(lastT_queue_lengths)
    price_vector = []
    # centralized price is total queue length at each time step
    # probably needs to be replaced as estimation of queue length at t + 1
    # based on q(t) sum of rates, and service rate
    # central_p = (1 / lambda_Switch) * (sum_ql + sum_session_rates
    #                                    - lambda_Switch)
    central_p = (sum_ql + sum_session_rates - lambda_Switch)
    price_vector.append(max(central_p, 0))

    for u in range(NumUsers):
        p_u = 0
        for session in userSessions[u]:
            # each user price is sum over queue lengths from their sessions
            # probably needs to be replaced also
            p_u += (lastT_queue_lengths[int(session - 1)]
                    + lastT_rates[int(session - 1)])
        p_u -= user_max_rates[u]
        # Scaling of price, currently by 1/lambda*_u
        # p_u = p_u / user_max_rates[u]
        price_vector.append(max(p_u, 0))

    return price_vector


# for now: assume log utility functions with weights all equal to constraints
# this performs gradient projection
def update_rates(NumUsers: int, NQs: int,
                 price_vector: list[int],
                 session_min_rates: list,
                 session_max_rates: list) -> list[float]:

    rates = []
    for s in range(NQs):
        user_pair = session_id_to_users(NumUsers, s + 1)

        p_us = 0
        for u in user_pair:
            p_us += price_vector[1 + u]

        price = price_vector[0] + p_us
        if price > 0:
            rate = max(session_min_rates[s], 1/price)
        else:
            # This value is meant to represent infinity -- just should be
            # larger than other scales in problem, represents 1/price when
            # price is zero
            rate = NQs
        rate = min(rate, session_max_rates[s])
        rates.append(rate)

    return rates


# should set up to also track rate recieved
def sim_QL_w_rate_feedback(NumUsers: int, H_num: int,
                           threshold: float,
                           user_max_rates: list,
                           session_min_rates: list,
                           gen_prob: float,
                           max_sched_per_q: int,
                           iters: int) -> Tuple[np.ndarray, np.ndarray,
                                                np.ndarray, list, list,
                                                list]:

    NQs = int(bc(NumUsers, 2))
    queue_lengths = np.zeros((NQs, iters))
    rate_track = np.zeros((NQs, iters))
    delivered = np.zeros((NQs, iters))
    schedule = np.zeros(NQs)

    # record user sessions
    userSessions = partition_sessions_by_user(NumUsers, NQs)
    # session max rates
    session_max_rates = [max_sched_per_q * gen_prob] * NQs
    # initial price vector (zero for empty queues)
    price_vector = [0] * (1 + NumUsers)

    # set initial requests
    rates = update_rates(NumUsers, NQs, price_vector, session_min_rates,
                         session_max_rates)

    rate_track[:, 0] = rates

    for x in range(iters):
        # Get submitted requests
        arrivals = gen_arrivals(NumUsers, rates)

        if x > 0:
            # based on prices, sources update rates
            price_vector = update_prices(NumUsers, userSessions,
                                         threshold, user_max_rates,
                                         queue_lengths[:, x - 1],
                                         rate_track[:, x - 1])
            rates = update_rates(NumUsers, NQs, price_vector,
                                 session_min_rates, session_max_rates)

            rate_track[:, x] = rates

            # for current schedule, do link gen
            schedule = model_probabilistic_link_gen(NumUsers,
                                                    gen_prob,
                                                    schedule)

            # track delivery of entangled pairs to sessions
            delivered[:, x] = schedule

            # Update queue lengths at x based on lengths at x-1, schedule
            # from x-1,
            # successful link gen at x, and arrivals at x
            # Essentially lengths by end of x
            queue_lengths[:, x] = (queue_lengths[:, x-1] + arrivals[:]
                                   - schedule[:])
        else:
            queue_lengths[:, x] = arrivals[:]

        schedule = flexible_schedule(H_num, NQs,
                                     queue_lengths[:, x],
                                     max_sched_per_q)

    return (queue_lengths, rate_track, delivered)


def plot_total_rates(rates: np.ndarray, NumUsers: int, H_num: int,
                     gen_prob: float, threshold: float, dist_fac: float,
                     iters: int, figname: str) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, p = {}, T = {}'.format(
                 NumUsers, H_num, gen_prob, threshold),
                 fontsize=28)
    av1 = (sum(rates[0, :]) / iters)
    ax1.plot(range(iters), rates[0, :], color=cmap(0),
             label='T - {}'.format(dist_fac * threshold))
    ax1.plot(range(iters), [av1] * iters, '--',
             color=cmap(inds[2]), label='{}'.format(round(av1, 3)))
    av2 = (sum(rates[1, :]) / iters)
    ax2.plot(range(iters), rates[1, :], color=cmap(inds[1]),
             label='T')
    ax2.plot(range(iters), [av2] * iters, '--',
             color=cmap(0), label='{}'.format(round(av2, 3)))
    av3 = sum(rates[2, :]) / iters
    ax3.plot(range(iters), rates[2, :], color=cmap(inds[2]),
             label='T + {}'.format(dist_fac * threshold))
    ax3.plot(range(iters), [av3] * iters, '--',
             color=cmap(0), label='{}'.format(round(av3, 3)))

    ax3.legend(fontsize=22, framealpha=0.6, loc=2)

    ax2.legend(fontsize=22, framealpha=0.6, loc=2)

    ax1.legend(fontsize=22, framealpha=0.6, loc=2)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_rate_profile(all_rates: List[np.ndarray], NumUsers: int,
                      H_num: int, gen_prob: float, threshold: float,
                      dist_fac: float, iters: int) -> None:

    cmap = plt.cm.get_cmap('plasma')
    NQs = int(bc(NumUsers, 2))
    inds = np.linspace(0, 0.95, NQs)
    numlabs = ['T - {}'.format((((dist_fac * threshold) // (1/1000)) / 1000)),
               'T',
               'T + {}'.format((((dist_fac * threshold) // (1/1000)) / 1000))]
    wordlabs = ['BelowT', 'AtT', 'AboveT']

    for x in range(3):
        plt.figure(figsize=(10, 8))
        for y in range(NQs):
            plt.plot(range(iters), all_rates[x][y, :], color=cmap(inds[y]))
            plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
            plt.title(numlabs[x])

            figname = '../Figures/AlgAdjust/RateProfile_{}_{}_{}'.format(
                      NumUsers, H_num, wordlabs[x])
            plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_delivery_rates(moving_avrgs: np.ndarray, avrg_delivered: list,
                        figname: str, iters: int, ptsInAvrg: int) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

    ax1.plot(range(iters), moving_avrgs[0], color=cmap(0),
             label='{} pt Avrg'.format(ptsInAvrg))
    ax1.plot(range(iters), [avrg_delivered[0]] * iters, '--',
             color=cmap(inds[2]),
             label='Avrg={}'.format(round(avrg_delivered[0], 3)))
    ax2.plot(range(iters), moving_avrgs[1], color=cmap(inds[1]),
             label='{} pt Avrg'.format(ptsInAvrg))
    ax2.plot(range(iters), [avrg_delivered[1]] * iters, '--',
             color=cmap(0),
             label='Avrg={}'.format(round(avrg_delivered[1], 3)))
    ax3.plot(range(iters), moving_avrgs[2], color=cmap(inds[2]),
             label='{} pt Avrg'.format(ptsInAvrg))
    ax3.plot(range(iters), [avrg_delivered[2]] * iters, '--',
             color=cmap(0),
             label='Avrg={}'.format(round(avrg_delivered[2], 3)))

    ax3.legend(fontsize=22, framealpha=0.4, loc=1)

    ax2.legend(fontsize=22, framealpha=0.4, loc=1)

    ax1.legend(fontsize=22, framealpha=0.4, loc=1)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.savefig(figname, dpi=300, bbox_inches='tight')

    return


def study_balance_near_threshold(NumUsers: int, H_num: int,
                                 user_max_rates: list,
                                 session_min_rates: list,
                                 gen_prob: float,
                                 max_sched_per_q: int,
                                 iters: int, dist_fac: float) -> None:

    threshold = ((H_num * gen_prob)
                 // (1/10000)) / 10000  # Truncate at 4th place

    (q1, rts1, d1) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            (1 - dist_fac) * (threshold),
                                            user_max_rates,
                                            session_min_rates,
                                            gen_prob,
                                            max_sched_per_q, iters)
    (q2, rts2, d2) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            threshold,
                                            user_max_rates,
                                            session_min_rates, gen_prob,
                                            max_sched_per_q, iters)
    (q3, rts3, d3) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            (1 + dist_fac) * threshold,
                                            user_max_rates,
                                            session_min_rates,
                                            gen_prob,
                                            max_sched_per_q, iters)

    Nexcl = 1000
    pltQStab = True
    if pltQStab:

        ql1, ql2, ql3 = np.zeros(iters - Nexcl), np.zeros(iters - Nexcl), \
                        np.zeros(iters - Nexcl)
        for x in range(Nexcl, iters):
            # Store total queue lengths
            ql1[x - Nexcl] = np.sum(q1[:, x], axis=0)
            ql2[x - Nexcl] = np.sum(q2[:, x], axis=0)
            ql3[x - Nexcl] = np.sum(q3[:, x], axis=0)

        p_whole = int(1000 * gen_prob)
        figname = '../Figures/AlgAdjust/QStab_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_queue_stability(ql1, ql2, ql3, NumUsers, H_num,
                             gen_prob, threshold, dist_fac, (iters - Nexcl),
                             figname)

    pltTotRts = True
    if pltTotRts:
        sum_rates = np.zeros((3, (iters - Nexcl)))
        for x in range(Nexcl, iters):
            sum_rates[0, x - Nexcl] = np.sum(rts1[:, x], axis=0)
            sum_rates[1, x - Nexcl] = np.sum(rts2[:, x], axis=0)
            sum_rates[2, x - Nexcl] = np.sum(rts3[:, x], axis=0)
        p_whole = int(100 * gen_prob)
        figname = '../Figures/AlgAdjust/RateTotals_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_total_rates(sum_rates, NumUsers, H_num, gen_prob, threshold,
                         dist_fac, (iters - Nexcl), figname)

    pltRtProfile = True
    if pltRtProfile:

        all_rates = [rts1[:, Nexcl:iters], rts2[:, Nexcl:iters],
                     rts3[:, Nexcl:iters]]

        plot_rate_profile(all_rates, NumUsers, H_num, gen_prob, threshold,
                          dist_fac, (iters - Nexcl))

    pltIdvQs = False
    if pltIdvQs:

        plot_individual_queues(q1[:, Nexcl:iters], q2[:, Nexcl:iters],
                               q3[:, Nexcl:iters], (iters - Nexcl), NumUsers,
                               H_num, dist_fac, threshold)

    # look at actually delivered rates (total)
    # look at in two ways:
    # 1. total average over course of simulation
    # 2. Moving average of x number of points
    pltDelivered = True
    if pltDelivered:

        ptsInAvrg = 100
        average_delivered = [(np.sum(d1) / iters),
                             (np.sum(d2) / iters),
                             (np.sum(d3) / iters)]

        avrgs = np.zeros((3, (iters - ptsInAvrg)))
        deliveries = [d1, d2, d3]
        for dv in range(3):
            for x in range(ptsInAvrg, iters):
                sum = 0
                for y in range(ptsInAvrg):
                    sum += np.sum(deliveries[dv][:, x - y])
                avrgs[dv, x - ptsInAvrg] = sum / ptsInAvrg
        figname = '../Figures/AlgAdjust/DeliveryRates_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_delivery_rates(avrgs, average_delivered,
                            figname, (iters - ptsInAvrg), ptsInAvrg)

    return

#  Write simulation to optimize the stepsize w/o scaling
#  want to record max fluctuations detected after first x steps have passed,
#  for x = 10, 100, 1000; want average queue backlog (discard 1000 steps)


NumUsers = 7
H_num = 2
NQs = int(bc(NumUsers, 2))
max_sched_per_q = 1
p_gen = 0.075
global_scale = 100
iters = 50000
dist_fac = 0.05
# Should relate to timescale of system
# One node can be involved in N-1 sessions
# per session a mx of p_gen ent generated per slot
# maybe one user can deal with a max of ((NQs - 1) / 2) * p_gen pair generated
# per slot, as example where user cutoffs are actually relevant
user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers
# user_max_rates = [H_num * p_gen] * NumUsers
# try user_max_rates set to NQs for case when they are not relevant
# user_max_rates = [NQs] * NQs
userSessions = partition_sessions_by_user(NumUsers, NQs)
session_min_rates = [p_gen / global_scale] * NQs

study_balance_near_threshold(NumUsers, H_num, user_max_rates,
                             session_min_rates, p_gen, max_sched_per_q,
                             iters, dist_fac)
