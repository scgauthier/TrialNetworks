import os
import shutil
import multiprocessing
import numpy as np
from scipy.special import binom as bc
from typing import Tuple
from random import random, sample
from math import floor, ceil

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


def session_id_to_users(NumUsers: int, session_id: int) -> list:

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
def partition_sessions_by_user(NumUsers: int,
                               NQs: int,
                               sessionMap: np.ndarray) -> np.ndarray:

    userSessions = np.zeros((NumUsers, NumUsers - 1))
    for x in range(NQs):
        min_entries = np.argmin(userSessions, axis=1)
        try:
            user_pair = session_id_to_users(NumUsers,
                                            int(sessionMap[x]) + 1)
        except IndexError:
            user_pair = session_id_to_users(NumUsers,
                                            int(sessionMap + 1))
        try:
            for u in user_pair:
                userSessions[u, min_entries[u]] = int(sessionMap[x]) + 1
        except IndexError:
            userSessions[u, min_entries[u]] = int(sessionMap) + 1

    return userSessions


def update_prices(NumUsers: int, userSessions: np.ndarray,
                  lambda_Switch: float, user_max_rates: list,
                  step_size: float,
                  central_scale: float,
                  sessionMap: np.ndarray,
                  lastT_queue_lengths: np.ndarray,
                  lastT_rates: np.ndarray) -> list:

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
            if session > 0:
                try:
                    mappedSession = int(np.where(
                                        sessionMap == (session - 1))[0][0])
                except IndexError:
                    mappedSession = int(sessionMap)
                # each user price is sum over queue lengths from their sessions
                # plus difference in sum of their session rates from user max
                p_u += (lastT_queue_lengths[int(mappedSession)]
                        + lastT_rates[int(mappedSession)])
        p_u -= user_max_rates[u]
        # Scaling of price, currently by 1/bar{lambda_u}
        p_u = p_u / user_max_rates[u]
        price_vector.append(max(p_u, 0))

    return price_vector


# for now: assume log utility functions with weights all equal to constraints
# this performs gradient projection
def update_rates(NumUsers: int, NQs: int,
                 price_vector: list,
                 sessionMap: np.ndarray,
                 session_min_rates: list,
                 session_max_rates: list) -> list:

    rates = []
    for s in range(NQs):
        try:
            mappedSession = int(sessionMap[s])
        except IndexError:
            mappedSession = int(sessionMap)
        user_pair = session_id_to_users(NumUsers, mappedSession + 1)

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


def vary_H(H_num: int, max_H: int, min_H: int) -> int:

    compar = random()
    if (compar > 0.5) and ((H_num + 1) <= max_H):
        H_num += 1
    elif (compar <= 0.5) and ((H_num - 1) >= min_H):
        H_num -= 1

    return H_num


# should set up to also track rate recieved
def sim_QL_w_rate_feedback(NumUsers: int,
                           params: dict,
                           trk_list: list,
                           run: int) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, list]:

    # local param copy
    loc_params = dict(params)
    # unpack params
    iters = loc_params['iters']

    H_num, p_gen = loc_params['H_num'], loc_params['p_gen']
    user_max_rates = loc_params['user_max_rates']
    session_min_rates = loc_params['session_min_rates']

    step_size = loc_params['step_size']
    max_sched_per_q = loc_params['max_sched_per_q']

    param_change = loc_params['param_change']

    max_H = H_num
    min_H = 1
    # min_H = max(1, H_num - 2)

    threshold = ((H_num * p_gen)
                 // (1/10000)) / 10000  # Truncate at 4th place

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/SeshMap.txt'
    NQs = ceil(int(bc(NumUsers, 2)) * params['sessionSamples'])
    sessionMap = np.loadtxt(fileName)
    queue_lengths = np.zeros((NQs, iters))
    rate_track = np.zeros((NQs, iters))
    delivered = np.zeros((NQs, iters))
    schedule = np.zeros(NQs)

    # record user sessions
    userSessions = partition_sessions_by_user(NumUsers, NQs, sessionMap)
    # session max rates
    session_max_rates = [max_sched_per_q * p_gen] * NQs
    # initial price vector (zero for empty queues)
    price_vector = [0] * (1 + NumUsers)

    # set initial requests
    rates = update_rates(NumUsers, NQs, price_vector, sessionMap,
                         session_min_rates, session_max_rates)

    rate_track[:, 0] = rates

    for x in range(iters):

        # Get submitted requests
        arrivals = gen_arrivals(NumUsers, rates, params)

        # if (x == 0) and (run == 0):
        #     trk_list.append(H_num)
        #     queue_lengths[:, x] = arrivals[:]

        if (x > 0):

            if param_change:

                change_key = loc_params['change_key']

                if (change_key == 'ChangeH') and (x in loc_params['indices']):

                    user_scale_factor = params['user_scale_factor']

                    if run > 0:
                        eval = np.where(loc_params['indices'] == x)[0][0]
                        loc_params['H_num'] = trk_list[eval]

                    else:
                        loc_params['H_num'] = vary_H(H_num, max_H, min_H)
                        trk_list.append(loc_params['H_num'])

                    H_num = loc_params['H_num']
                    threshold = ((H_num * p_gen)
                                 // (1/10000)) / 10000  # Truncate at 4th place
                    loc_params['central_scale'] = user_scale_factor / threshold

            # based on prices, sources update rates
            price_vector = update_prices(NumUsers, userSessions,
                                         threshold, user_max_rates,
                                         step_size,
                                         loc_params['central_scale'],
                                         sessionMap,
                                         queue_lengths[:, x - 1],
                                         rate_track[:, x - 1])
            rates = update_rates(NumUsers, NQs, price_vector, sessionMap,
                                 session_min_rates, session_max_rates)

            rate_track[:, x] = rates

            # for current schedule, do link gen
            schedule = model_probabilistic_link_gen(NumUsers,
                                                    p_gen,
                                                    params,
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

    return (queue_lengths, rate_track, delivered, trk_list)


def study_balance_near_threshold(NumUsers: int, H_num: int,
                                 user_max_rates: list,
                                 session_min_rates: list,
                                 step_size: float,
                                 central_scale: float,
                                 p_gen: float,
                                 max_sched_per_q: int,
                                 iters: int, dist_fac: float) -> None:

    threshold = ((H_num * p_gen)
                 // (1/10000)) / 10000  # Truncate at 4th place

    params = load_params(NumUsers)
    params['p_gen'] = (1 - dist_fac) * p_gen
    (q1, rts1, d1) = sim_QL_w_rate_feedback(NumUsers, params, [], 0)
    params['p_gen'] = p_gen
    (q2, rts2, d2) = sim_QL_w_rate_feedback(NumUsers, params, [], 0)
    params['p_gen'] = (1 + dist_fac) * p_gen
    (q3, rts3, d3) = sim_QL_w_rate_feedback(NumUsers, params, [], 0)

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

        p_whole = int(1000 * p_gen)
        figname = '../Figures/AlgAdjust/QStab_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_queue_stability(ql1, ql2, ql3, NumUsers, H_num,
                             p_gen, threshold, dist_fac, (iters - Nexcl),
                             figname)

    pltTotRts = True
    if pltTotRts:
        sum_rates = np.zeros((3, (iters - Nexcl)))
        for x in range(Nexcl, iters):
            sum_rates[0, x - Nexcl] = np.sum(rts1[:, x], axis=0)
            sum_rates[1, x - Nexcl] = np.sum(rts2[:, x], axis=0)
            sum_rates[2, x - Nexcl] = np.sum(rts3[:, x], axis=0)
        p_whole = int(1000 * p_gen)
        figname = '../Figures/AlgAdjust/RateTotals_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_total_rates(sum_rates, NumUsers, H_num, p_gen, threshold,
                         dist_fac, (iters - Nexcl), figname, True)

    pltRtProfile = False
    if pltRtProfile:

        all_rates = [rts1[:, Nexcl:iters], rts2[:, Nexcl:iters],
                     rts3[:, Nexcl:iters]]

        tag = 'UniformFixed'
        plot_rate_profile(all_rates, NumUsers, H_num, p_gen, threshold,
                          dist_fac, (iters - Nexcl), 1, False, tag)

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
        p_whole = int(1000 * p_gen)
        figname = '../Figures/AlgAdjust/DeliveryRates_LR_{}_{}_{}'.format(
                NumUsers, H_num, p_whole)

        plot_delivery_rates(avrgs, average_delivered,
                            figname, (iters - ptsInAvrg), ptsInAvrg, True)

    return


def record_NumUsers(NumUsers: int, params: dict) -> None:
    params['NumUsers'] = NumUsers
    # write number of users to param file
    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/paramLog.txt'
    afile = open(fileName, 'a')
    afile.write('NumUsers: {}\n'.format(NumUsers))
    afile.close()
    return


def record_trk_list(trk_list: list, params: dict) -> None:

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/trkList.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, np.array(trk_list))
        afile.close()


def record_midProcess(sum_rates: np.ndarray,
                      rate_requests: np.ndarray,
                      sum_delivery: np.ndarray,
                      max_rates: np.ndarray,
                      min_rates: np.ndarray,
                      params: dict,
                      run: int) -> None:
    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/SR'
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    fileName = dirName + '/{}.txt'.format(run)
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, sum_rates)
        afile.close()

    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/DR'
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    fileName = dirName + '/{}.txt'.format(run)
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, sum_delivery)
        afile.close()

    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/MR'
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    fileName = dirName + '/{}.txt'.format(run)
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, max_rates)
        afile.close()

    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/mr'
    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    fileName = dirName + '/{}.txt'.format(run)
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, min_rates)
        afile.close()

    # dirName = '../DataOutput/{}'.format(params['timeStr']) + '/RP'
    # if not os.path.isdir(dirName):
    #     os.mkdir(dirName)
    # fileName = dirName + '/{}.txt'.format(run)
    # if not os.path.isfile(fileName):
    #     afile = open(fileName, 'w')
    #     for row in rate_requests:
    #         np.savetxt(afile, row)
    #     afile.close()


def record_dataSets(average_requests: np.ndarray,
                    rate_profile: np.ndarray,
                    params: dict) -> None:

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/AvReq.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, average_requests)
        afile.close()

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/RtProf.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        for row in rate_profile:
            np.savetxt(afile, row)
        afile.close()

    return


def record_AvDataSet(average_requests: np.ndarray,
                     params: dict) -> None:

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/AvReq.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, average_requests)
        afile.close()

    return


def record_AvDelSet(average_delivery: np.ndarray,
                    params: dict) -> None:

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/AvDel.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, average_delivery)
        afile.close()

    return


def record_MaxMinDataSet(max_requests: np.ndarray,
                         min_requests: np.ndarray,
                         params: dict) -> None:

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/MaxReq.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, max_requests)
        afile.close()

    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/minReq.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, min_requests)
        afile.close()

    return


def record_session_map(NumUsers: int, params: dict) -> None:

    NQs = int(bc(NumUsers, 2))
    nSamples = ceil(params['sessionSamples'] * NQs)
    session_map = sample(range(NQs), nSamples)
    fileName = '../DataOutput/{}'.format(params['timeStr']) + '/SeshMap.txt'
    if not os.path.isfile(fileName):
        afile = open(fileName, 'w')
        np.savetxt(afile, session_map)
        afile.close()

    return


def get_runAvrgs(param_tuple: tuple) -> Tuple[np.ndarray, np.ndarray]:

    params, trk_list, run = param_tuple
    NumUsers, iters = params['NumUsers'], params['iters']
    Nexcl = params['Nexcl']
    study_delivery = params['study_delivery']

    (queues,
     requested_rates,
     delivered_rates,
     trk_list) = sim_QL_w_rate_feedback(NumUsers,
                                        params,
                                        trk_list,
                                        run)

    sum_rates = np.zeros(iters)
    sum_delivery = np.zeros(iters)
    max_rates = np.zeros(iters)
    min_rates = np.zeros(iters)
    for x in range(Nexcl, iters):
        sum_rates[x - Nexcl] = np.sum(requested_rates[:, x], axis=0)
        if study_delivery:
            sum_delivery[x - Nexcl] = np.sum(delivered_rates[:, x], axis=0)
        max_rates[x - Nexcl] = np.max(requested_rates[:, x])
        min_rates[x - Nexcl] = np.min(requested_rates[:, x])
    record_midProcess(sum_rates, requested_rates,
                      sum_delivery,
                      max_rates, min_rates,
                      params, run)

    return


# use distance fac for plotting guidelines?
def study_algorithm(NumUsers: int,
                    params: dict) -> None:

    # unpack params
    iters, runs = params['iters'], params['runs']
    Nexcl = params['Nexcl']
    study_delivery = params['study_delivery']
    # NQs = int(bc(NumUsers, 2))

    trk_list = []
    record_session_map(NumUsers, params)

    # Handle run 0 separately, set up trk_list
    (queues,
     requested_rates,
     delivered_rates,
     trk_list) = sim_QL_w_rate_feedback(NumUsers,
                                        params,
                                        trk_list,
                                        0)
    sum_rates = np.zeros(iters)
    sum_delivery = np.zeros(iters)
    max_rates = np.zeros(iters)
    min_rates = np.zeros(iters)
    for x in range(Nexcl, iters):
        sum_rates[x - Nexcl] = np.sum(requested_rates[:, x], axis=0)
        if study_delivery:
            sum_delivery[x-Nexcl] = np.sum(delivered_rates[:, x], axis=0)
        max_rates[x - Nexcl] = np.max(requested_rates[:, x])
        min_rates[x - Nexcl] = np.min(requested_rates[:, x])
    record_midProcess(sum_rates, requested_rates,
                      sum_delivery,
                      max_rates, min_rates,
                      params, 0)

    # set up multiprocessing -- create pool of worker processes
    num_cpus = min(multiprocessing.cpu_count(), 30)
    mypool = multiprocessing.Pool(num_cpus)
    # update fidelities
    mypool.map(get_runAvrgs, [(params,
                              trk_list,
                              run) for run in range(1, runs)])

    average_requests = np.zeros(iters)
    if study_delivery:
        average_delivery = np.zeros(iters)
    max_requests = np.zeros(iters)
    min_requests = np.zeros(iters)
    # rate_profile = np.zeros((int(bc(NumUsers, 2)), iters))

    for run in range(runs):
        dirName = '../DataOutput/{}'.format(params['timeStr']) + '/SR'
        fileName = dirName + '/{}.txt'.format(run)
        if os.path.isfile(fileName):
            average_requests += np.loadtxt(fileName)
        dirName = '../DataOutput/{}'.format(params['timeStr']) + '/DR'
        fileName = dirName + '/{}.txt'.format(run)
        if os.path.isfile(fileName):
            average_delivery += np.loadtxt(fileName)
        # dirName = '../DataOutput/{}'.format(params['timeStr']) + '/RP'
        # fileName = dirName + '/{}.txt'.format(run)
        # if os.path.isfile(fileName):
        #     rate_profile += np.loadtxt(fileName).reshape(NQs, iters)
        dirName = '../DataOutput/{}'.format(params['timeStr']) + '/MR'
        fileName = dirName + '/{}.txt'.format(run)
        if os.path.isfile(fileName):
            max_requests += np.loadtxt(fileName)
        dirName = '../DataOutput/{}'.format(params['timeStr']) + '/mr'
        fileName = dirName + '/{}.txt'.format(run)
        if os.path.isfile(fileName):
            min_requests += np.loadtxt(fileName)
    average_requests *= (1 / runs)
    if study_delivery:
        average_delivery *= (1 / runs)
    max_requests *= (1 / runs)
    min_requests *= (1 / runs)
    # rate_profile *= (1 / runs)

    # Remove un-needed intermediaries (clear up space)
    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/SR'
    shutil.rmtree(dirName, ignore_errors=True)
    if study_delivery:
        dirName = '../DataOutput/{}'.format(params['timeStr']) + '/DR'
        shutil.rmtree(dirName, ignore_errors=True)

    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/MR'
    shutil.rmtree(dirName, ignore_errors=True)

    dirName = '../DataOutput/{}'.format(params['timeStr']) + '/mr'
    shutil.rmtree(dirName, ignore_errors=True)
    # dirName = '../DataOutput/{}'.format(params['timeStr']) + '/RP'
    # shutil.rmtree(dirName, ignore_errors=True)

    # record_dataSets(average_requests,
    #                 rate_profile,
    #                 params)
    record_AvDataSet(average_requests,
                     params)
    if study_delivery:
        record_AvDelSet(average_delivery,
                        params)
    record_MaxMinDataSet(max_requests, min_requests,
                         params)

    dirName = '../DataOutput/{}'.format(params['timeStr'])
    fgnm = dirName + '/AvReq.txt'

    pltTotRts = False
    if pltTotRts:
        plot_total_rates(average_requests, NumUsers, params,
                         trk_list, fgnm, False)

    record_trk_list(trk_list, params)

    return

#Phils input xoxo
#def papi_test(papi):
#if papi == strong:
    #print("You're a loser papi")
#else:
    #print("Pull harder papi")
#return


def load_user_max_rates(NumUsers: int,
                        p_gen: float,
                        NQs: int,
                        max_sched_per_q: int,
                        keyword: str) -> list:
    user_max_rates = [p_gen * max_sched_per_q] * NumUsers

    if keyword == 'uniformVeryHigh':
        if NQs > 1:
            user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers
        else:
            user_max_rates = [((NQs) / 2) * p_gen] * NumUsers
    elif keyword == 'uniformSessionMax':
        user_max_rates = [p_gen * max_sched_per_q] * NumUsers
    elif keyword == 'halfUniformSessionMax':
        user_max_rates = [p_gen * max_sched_per_q / 2] * NumUsers
    elif keyword == 'singleNonUniformSessionMax':
        first_partition = sample(range(NumUsers), floor(NumUsers / 4))
        # random N/4 subset has max user = half max session
        for x in first_partition:
            user_max_rates[x] = (p_gen * max_sched_per_q) / 2
    elif keyword == 'doubleNonUniformSessionMax':
        first_partition = sample(range(NumUsers), floor(NumUsers / 4))
        remains = list(range(NumUsers))
        # random subset have max user = half max session
        for x in first_partition:
            user_max_rates[x] = (p_gen * max_sched_per_q) / 2
            remains.remove(x)
        # another random subset have max user  = 1.5 * max session
        for x in sample(remains, floor(NumUsers / 4)):
            user_max_rates[x] = (p_gen * max_sched_per_q) * 1.5
    return user_max_rates
