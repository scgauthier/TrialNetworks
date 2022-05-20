import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from typing import Tuple, List
from math import sqrt

from ToySwitch import prep_dist, gen_arrivals, model_probabilistic_link_gen
from ToySwitch import flexible_schedule, weighted_flex_schedule
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


def define_tolerances(rates: list) -> list:

    tolerances = []
    for rt in rates:
        # temp tolerances are 20% of waiting time
        # tolerances.append(0.2 * 1/rt)
        tolerances.append(1 * (1 / sqrt(rt * (1 - rt))))
    return tolerances


# Note: I think additive increase multiplicative decrease can be
# accomplished by using some fraction of p_gen as the additive increase.
# Note that p_gen is a local parameter to each node.
def decrease_rate(rate: float) -> float:
    # return (0.1 * rate**2)
    return (0.01 * rate)


def increase_rate(rate: float, p_gen: float) -> float:
    # return (0.1 * rate**(1/4))
    return (0.005 * p_gen)


# service times = most recent service time
# waiting times = waiting time most recent = interval observed b/w service)
def adjust_rates(rates: list, time: int,
                 genPs: list, max_sched_per_q: int,
                 service_times: np.ndarray,
                 waiting_times: np.ndarray,
                 previous_WT: np.ndarray) -> list:

    tolerances = define_tolerances(rates)

    for x in list(range(len(rates))):
        global_scale = 10 * 10
        # av_wt = ((waiting_times[x] + previous_WT[x]) / 2)
        av_wt = ((0.55 * waiting_times[x])
                 + (0.45 * previous_WT[x]))
        cap = genPs[x] * max_sched_per_q
        min_rate = genPs[x] / global_scale

        if ((service_times[x] == time) and (time > 0)):
            upper_bound = (1 / rates[x]) + tolerances[x]
            lower_bound = (1 / rates[x] + (tolerances[x] / 10))
            # bound = (1 / rates[x]) + tolerances[x]

            if (av_wt > upper_bound):
                rates[x] -= decrease_rate(rates[x])
                rates[x] = max(rates[x], min_rate)

            # increase if possible, not to above cap
            elif (av_wt <= lower_bound):
                rates[x] = min(rates[x] + increase_rate(rates[x], genPs[x]),
                               cap)
                rates[x] = max(rates[x], min_rate)

        else:
            scale_tol = 1.5
            idle_time = time - service_times[x]
            upper_bound = (1 / rates[x]) + (scale_tol * tolerances[x])

            if idle_time > upper_bound:
                rates[x] -= decrease_rate(rates[x])
                rates[x] = max(rates[x], min_rate)

    return rates


def sim_QL_w_rate_feedback(NumUsers: int, H_num: int,
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
    observed_times_perQ = np.zeros((5, NQs))
    waiting_dist_max = []
    waiting_dist_min = []
    avrg_rates = np.zeros(NQs)
    service_times = np.zeros(NQs)
    rate_track = np.zeros((NQs, iters))
    schedule = np.zeros(NQs)
    marked = np.zeros(NQs)

    # for now, all use same p_gen
    genPs = NQs * [gen_prob]
    # set initial requests
    probs = prep_dist(NumUsers, H_num, prob_param, pDist, max_sched_per_q)
    # track index of initial max and min requests
    ratemax, ratemin = probs.index(max(probs)), probs.index(min(probs))
    rate_track[:, 0] = probs

    for x in range(iters):
        # Get submitted requests
        arrivals = gen_arrivals(NumUsers, pDist, probs)
        # Update submission times
        for y in np.nonzero(arrivals):
            submission_times[y, x] = arrivals[y]

        # based on prev link gen, sources update rates
        # essentially assumes 1 timestep delay b/w state gen and rate changes
        probs = adjust_rates(probs, (x - 1), genPs, max_sched_per_q,
                             observed_times_perQ[2, :],
                             observed_times_perQ[3, :],
                             observed_times_perQ[4, :])
        rate_track[:, x] = probs
        # print(probs, " Sum: ", sum(probs))

        # for current schedule, do link gen
        schedule, marked = model_probabilistic_link_gen(NumUsers,
                                                        gen_prob,
                                                        schedule)

        # Update service times:
        # 2 notions: waiting_times: time requests wait in q b/w sub and served
        #          : observed_times: time between successive served requests
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
                # store previous waiting times
                observed_times_perQ[4, y] = observed_times_perQ[3, y]
                observed_times_perQ[3, y] = service_int  # update waiting

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
                                         marked,
                                         max_sched_per_q)
        elif sched_type == 'weighted':
            schedule = weighted_flex_schedule(H_num, NQs,
                                              probs,
                                              queue_lengths[:, x],
                                              marked,
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
            probs, waiting_dist_max, waiting_dist_min, rate_track)


def plot_total_rates(rates: np.ndarray, NumUsers: int, H_num: int,
                     gen_prob: float, threshold: float, dist_fac: float,
                     iters: int, figname: str) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
    fig.suptitle('N = {}, H = {}, p = {}, T = {}'.format(
                 NumUsers, H_num, gen_prob, threshold),
                 fontsize=28)
    ax1.plot(range(iters), rates[0, :], color=cmap(0),
             label='T - {}'.format(dist_fac * threshold))
    ax2.plot(range(iters), rates[1, :], color=cmap(inds[1]),
             label='T')
    ax3.plot(range(iters), rates[2, :], color=cmap(inds[2]),
             label='T + {}'.format(dist_fac * threshold))

    ax3.legend(fontsize=22, framealpha=0.6, loc=2)

    ax2.legend(fontsize=22, framealpha=0.6, loc=2)

    ax1.legend(fontsize=22, framealpha=0.6, loc=2)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_rate_profile(all_rates: List[np.ndarray], NumUsers: int,
                      H_num: int, gen_prob: float, threshold: float,
                      dist_fac: float, sched_type: str,
                      iters: int) -> None:

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

            figname = '../Figures/PairRequest/RateProfile_{}_{}_{}_{}'.format(
                      NumUsers, H_num, sched_type, wordlabs[x])
            plt.savefig(figname, dpi=300, bbox_inches='tight')


def study_balance_near_threshold(NumUsers: int, H_num: int,
                                 pDist: str, gen_prob: float,
                                 sched_type: str, max_sched_per_q: int,
                                 iters: int,
                                 dist_fac: float) -> None:

    threshold = ((H_num * gen_prob)
                 // (1/10000)) / 10000  # Truncate at 4th place

    (q1, wt1, rt1, rr1,
     waitMax1, waitMin1,
     rate_track_1) = sim_QL_w_rate_feedback(NumUsers, H_num, pDist,
                                            (1 - dist_fac) * (threshold),
                                            gen_prob,
                                            sched_type,
                                            max_sched_per_q, iters)
    (q2, wt2, rt2, rr2,
     waitMax2, waitMin2,
     rate_track_2) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                            pDist, threshold, gen_prob,
                                            sched_type,
                                            max_sched_per_q, iters)
    (q3, wt3, rt3, rr3,
     waitMax3, waitMin3,
     rate_track_3) = sim_QL_w_rate_feedback(NumUsers, H_num, pDist,
                                            (1 + dist_fac) * threshold,
                                            gen_prob,
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
        figname = '../Figures/PairRequest/QStab_LR_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, p_whole, pDist, sched_type)

        plot_queue_stability(ql1, ql2, ql3, NumUsers, H_num,
                             gen_prob, threshold, dist_fac, iters, figname)

    pltTotRts = True
    if pltTotRts:
        rate_input = np.zeros((3, iters))
        for x in range(iters):
            rate_input[0, x] = np.sum(rate_track_1[:, x], axis=0)
            rate_input[1, x] = np.sum(rate_track_2[:, x], axis=0)
            rate_input[2, x] = np.sum(rate_track_3[:, x], axis=0)
        p_whole = int(100 * gen_prob)
        figname = '../Figures/PairRequest/RateTotals_LR_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, p_whole, pDist, sched_type)

        plot_total_rates(rate_input, NumUsers, H_num, gen_prob, threshold,
                         dist_fac, iters, figname)

    pltRtProfile = True
    if pltRtProfile:

        all_rates = [rate_track_1, rate_track_2, rate_track_3]

        plot_rate_profile(all_rates, NumUsers, H_num, gen_prob, threshold,
                          dist_fac, sched_type, iters)

    pltWTD = False
    if pltWTD:
        figname = '../Figures/PairRequest/LR_WTD_{}_{}_{}_{}_{}'.format(
                NumUsers, H_num, sched_type,
                round(1/max(rr1)), round(1/min(rr1)))
        plot_waiting_time_dists(waitMax1, waitMin1, rt1,
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


# study_balance_near_threshold(4, 2, 'uBin', 0.75,
#                              'base', 1, 10000, 0.05)
