import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import binom
from random import random, shuffle
from typing import Tuple, List
from itertools import combinations, combinations_with_replacement

from ToySwitch import prep_dist, gen_arrivals, model_probabilistic_link_gen
from ToySwitch import flexible_shortcut, weighted_flex_shortcut

from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)

# Base simulation off of sim in ToySwitch, but allow rates to vary.
# Also try to allow using either scheduling mode from beginning

def define_tolerances(rates: list) -> list

    tolerances = []
    for rt in rates:
        # temp tolerances are 50% of waiting time
        tolerances.append(0.5 * 1/rt)
    return tolerances


def decrease_rate(rate: float) -> float:
    return rate**2


def increase_rate(rate: float) -> float:
    return (0.1 * rate**(1/4))

# service times = most recent service time
# waiting times = waiting time most recent = interval observed b/w service)
def adjust_rates(rates: list, time: int,
                 service_times: np.ndarray,
                 waiting_times: np.ndarray,
                 service_flags: List[str]) -> list:

    tolerances = define_tolerances(rates)

    for x in range(rates):
        if service_flags[x] == 'pg':
            upper_bound = (1 / rates[x]) + tolerances[x]
            lower_bound = (1 / rates[x])

            if (waiting_times[x] > upper_bound):
                rates[x] = decrease_rate(rates[x])
            elif (waiting_times[x] <= lower_bound):
                rates[x] = increase_rate(rates[x])

        elif service_flags[x] == 'idle':
            scale_tol = 2
            idle_time = time - service_times[x]
            upper_bound = (1 / rates[x]) + (scale_tol * tolerances[x])

            if idle_time > upper_bound:
                rates[x] = decrease[x]

    return rates






def sim_QL_w_rate_feedback(NumUsers: int, H_num: int,
                           max_subs: int,
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
    observed_times_perQ = np.zeros((4, NQs))
    waiting_dist_max = []
    waiting_dist_min = []
    avrg_rates = np.zeros(NQs)
    service_times = np.zeros(NQs)
    schedule = np.zeros(NQs)
    marked = np.zeros(NQs)

    # set initial requests
    probs = prep_dist(NumUsers, H_num, prob_param, pDist, max_sched_per_q)
    # track index of initial max and min requests
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
                observed_times_perQ[3, y] = service_int # update waiting

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
                                         'rq',
                                         max_sched_per_q)
        elif sched_type == 'weighted':
            schedule = weighted_flex_shortcut(H_num, NQs,
                                              probs,
                                              queue_lengths[:, x],
                                              marked,
                                              'rq',
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
