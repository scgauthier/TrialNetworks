import numpy as np
from scipy.special import binom as bc
from typing import Tuple
from random import random

from ToySwitch import gen_arrivals, model_probabilistic_link_gen
from ToySwitch import flexible_schedule
from ToySwitch import plot_queue_stability, plot_individual_queues

from PlottingFcns import plot_total_rates, plot_rate_profile
from PlottingFcns import plot_delivery_rates
# from ToySwitch import plot_waiting_time_dists

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
                  step_size: float,
                  central_scale: float,
                  lastT_queue_lengths: np.ndarray,
                  lastT_rates: np.ndarray) -> list[int]:

    sum_session_rates = np.sum(lastT_rates)
    sum_ql = np.sum(lastT_queue_lengths)
    price_vector = []
    # centralized price is total queue length at each time step
    # probably needs to be replaced as estimation of queue length at t + 1
    # based on q(t) sum of rates, and service rate
    central_p = central_scale * (sum_ql
                                 + (step_size * (sum_session_rates
                                    - lambda_Switch)))
    # central_p = (sum_ql + sum_session_rates - lambda_Switch)
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
        p_u = p_u / user_max_rates[u]
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


def vary_H(params: Tuple, max_H: int, min_H: int) -> Tuple:

    H_num = params[0]
    compar = random()
    if (compar > 0.5) and ((H_num + 1) <= max_H):
        params[0] = H_num + 1
    elif (compar <= 0.5) and ((H_num - 1) >= min_H):
        params[0] = H_num - 1

    return params


# should set up to also track rate recieved
def sim_QL_w_rate_feedback(NumUsers: int,
                           params: Tuple,
                           max_sched_per_q: int,
                           iters: int,
                           param_change: bool) -> Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, list,
                                                        list, list]:

    # unpack param tuple
    H_num, gen_prob, user_max_rates, session_min_rates = params[:4]
    step_size, central_scale = params[4:]

    max_H = H_num + 1
    min_H = H_num - 1

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

            if param_change:
                change_key = params[6]

                if (change_key == 'ChangeH') and (x in params[7]):
                    vary_H(params, max_H, min_H)
                # unpack param tuple
                # repeating to allow changes between iters
                H_num, gen_prob, user_max_rates, session_min_rates = params[:4]
                step_size, central_scale = params[4:]

                threshold = ((H_num * gen_prob)
                             // (1/10000)) / 10000  # Truncate at 4th place

            # based on prices, sources update rates
            price_vector = update_prices(NumUsers, userSessions,
                                         threshold, user_max_rates,
                                         step_size,
                                         central_scale,
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


def study_balance_near_threshold(NumUsers: int, H_num: int,
                                 user_max_rates: list,
                                 session_min_rates: list,
                                 step_size: float,
                                 central_scale: float,
                                 gen_prob: float,
                                 max_sched_per_q: int,
                                 iters: int, dist_fac: float) -> None:

    threshold = ((H_num * gen_prob)
                 // (1/10000)) / 10000  # Truncate at 4th place

    params1 = [H_num, gen_prob, user_max_rates,
               session_min_rates, step_size,
               central_scale]
    (q1, rts1, d1) = sim_QL_w_rate_feedback(NumUsers, params1,
                                            max_sched_per_q, iters)
    (q2, rts2, d2) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            threshold,
                                            user_max_rates,
                                            session_min_rates,
                                            step_size,
                                            central_scale,
                                            max_sched_per_q, iters)
    (q3, rts3, d3) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            (1 + dist_fac) * threshold,
                                            user_max_rates,
                                            session_min_rates,
                                            step_size,
                                            central_scale,
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
        p_whole = int(1000 * gen_prob)
        figname = '../Figures/AlgAdjust/RateTotals_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_total_rates(sum_rates, NumUsers, H_num, gen_prob, threshold,
                         dist_fac, (iters - Nexcl), figname, True)

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
        p_whole = int(1000 * gen_prob)
        figname = '../Figures/AlgAdjust/DeliveryRates_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_delivery_rates(avrgs, average_delivered,
                            figname, (iters - ptsInAvrg), ptsInAvrg, True)

    return


# use distance fac for plotting guidelines?
def study_algorithm(NumUsers: int,
                    H_num: int,
                    user_max_rates: list,
                    session_min_rates: list,
                    step_size: float,
                    central_scale: float,
                    gen_prob: float,
                    max_sched_per_q: int,
                    iters: int, runs: int,
                    dist_fac: float,
                    param_change: bool) -> None:

    threshold = ((H_num * gen_prob)
                 // (1/10000)) / 10000  # Truncate at 4th place
    # Nexcl = 1000
    Nexcl = 0
    average_requests = np.zeros(iters)
    if param_change:
        change_key = 'ChangeH'
        indices = np.linspace(iters/10, iters, 10)
        params = [H_num, gen_prob, user_max_rates, session_min_rates,
                  step_size, central_scale, change_key, indices]
    else:
        params = [H_num, gen_prob, user_max_rates, session_min_rates,
                  step_size, central_scale]

    for run in range(runs):
        if (run % 10) == 0:
            print(run)
        (queues,
         requested_rates,
         delivered_rates) = sim_QL_w_rate_feedback(NumUsers, params,
                                                   max_sched_per_q, iters,
                                                   param_change)
        sum_rates = np.zeros(iters)
        for x in range(Nexcl, iters):
            sum_rates[x - Nexcl] = np.sum(requested_rates[:, x], axis=0)
        average_requests += sum_rates

    average_requests *= (1 / runs)

    p_whole = int(1000 * gen_prob)
    figname = '../Figures/AlgAdjust/AvRequestRates_{}_{}_{}_{}_{}'.format(
            NumUsers, H_num, p_whole, runs, int(100 * dist_fac))

    pltTotRts = True
    if pltTotRts:
        plot_total_rates(average_requests, NumUsers, H_num,
                         gen_prob, threshold, dist_fac,
                         iters, figname, False)

    pltRtProfile = True
    tag = 'UniformFixed'
    if pltRtProfile:
        plot_rate_profile(requested_rates[:, Nexcl:], NumUsers, H_num,
                          gen_prob, threshold,
                          dist_fac, (iters - Nexcl), False, tag)
    return


NumUsers = 4
H_num = 2
max_sched_per_q = 1
NQs = int(bc(NumUsers, 2))
p_gen = 0.05
global_scale = 1000
iters = 10001
runs = 1000
dist_fac = 0.01
# step_size = 1
lambda_Switch = H_num * p_gen
# Should relate to timescale of system
# One node can be involved in N-1 sessions
# per session a mx of p_gen ent generated per slot
# maybe one user can deal with a max of ((NQs - 1) / 2) * p_gen pair generated
# per slot, as example where user cutoffs are actually relevant
# user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers
user_max_rates = [H_num * p_gen] * NumUsers
# try user_max_rates set to NQs for case when they are not relevant
# user_max_rates = [NQs] * NQs
session_min_rates = [p_gen / global_scale] * NQs
param_change = False

# study_balance_near_threshold(NumUsers, H_num, user_max_rates,
#                              session_min_rates, step_size,
#                              p_gen, max_sched_per_q,
#                              iters, dist_fac)
study_algorithm(NumUsers, H_num, user_max_rates, session_min_rates,
                1, 1/lambda_Switch, p_gen, max_sched_per_q,
                iters, runs, dist_fac, param_change)
