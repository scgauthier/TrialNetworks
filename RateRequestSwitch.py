import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import norm, bernoulli
from random import random, sample
from math import sqrt
from time import localtime, strftime
from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)


# Parameters:
# -change rate: stipulates how often a single (unlocked) queue changes
#               it's declared rate
# mean_rate: the average rate declared by any queue
# rate_var: variance of declared rates
def F1_rate_declare(change_rate: float,
                    mean_rate: float,
                    rate_var: float,
                    time: int,
                    delay: float,
                    rates: np.ndarray,
                    change_locked: np.ndarray) -> list:

    rate_std = sqrt(rate_var)
    NQs = np.shape(rates)[0]
    up_rates = np.copy(rates)
    alterable = []
    for x in range(NQs):
        if (time >= (change_locked[x, 0] + delay)):
            alterable.append(x)
    for x in alterable:
        # if random() < (change_rate / NQs):  # Change rate equally distributed
        if random() < change_rate:
            draw = -1
            while draw < 0:
                draw = norm.rvs(mean_rate, rate_std, size=1)[0]
            up_rates[x, 0] = draw
            change_locked[x, 0] = time
    return up_rates, change_locked


def F1_schedule(declared_rates: np.ndarray,
                achieved_rates: np.ndarray,
                H_num: int,
                time: int,
                delay: int,
                locked: np.ndarray,
                schedule: np.ndarray,
                normalized=False) -> list:

    NQs = np.shape(declared_rates)[0]
    new_schedule = np.copy(schedule)

    # randomly schedule if nothing achieved yet
    if np.sum(achieved_rates) == 0:
        for x in sample(range(NQs), H_num):
            new_schedule[x, 0] = 1
            locked[x, 0] = time

    else:

        # should be +ve if not over achieving
        diffs = list((declared_rates - achieved_rates).flatten())

        # keep scheduled if within gen_window time:
        fixed = NQs
        for x in range(NQs):
            if time >= locked[x, 0] + delay:
                new_schedule[x, 0] = 0
                fixed -= 1
            else:
                diffs[x] = -1
        H_eff = H_num - fixed

        arranged_diffs = sorted(diffs)
        arranged_diffs.reverse()

        for x in range(H_eff):
            # Over blocking
            # if arranged_diffs[x] > 0:
            #     ind = diffs.index(arranged_diffs[x])
            #     new_schedule[ind, 0] = 1
            #     locked[ind, 0] = time

            # Not over blocking
            ind = diffs.index(arranged_diffs[x])
            new_schedule[ind, 0] = 1
            locked[ind, 0] = time

    return new_schedule, locked


# Can add blocking feature to model inability to gen at back to back times
def pair_gen(p_gen: float,
             schedule: np.ndarray
             ) -> np.ndarray:

    NQs = np.shape(schedule)[0]
    generated = np.zeros((NQs, 1))

    for x in np.nonzero(schedule[:, 0])[0]:
        generated[x, 0] = bernoulli.rvs(p_gen, size=1)[0]

    return generated


# For all times greater than 0
def rate_updates(ach_rates: np.ndarray,
                 generated: np.ndarray,
                 time: int) -> np.ndarray:
    for x in range(np.shape(generated)[0]):
        ach_rates[x, 0] = ((ach_rates[x, 0] * (time - 1))
                           + generated[x, 0]) / time  # New rate
    return ach_rates


# All rates start at just less than mean_rate
# Over-serves any rate that is already greater than demanded
def simulate_F1_service(NumUsers: int,
                        H_num: int,
                        p_gen: float,
                        gen_window: int,
                        flux_delay: int,
                        change_rate: float,
                        mean_rate: float,
                        rate_var: float,
                        runtime: int) -> np.ndarray:

    NQs = int(bc(NumUsers, 2))
    schedule = np.zeros((NQs, 1))
    dec_rates = np.ones((NQs, 1)) * (mean_rate - 0.01)
    ach_rates = np.zeros((NQs, 1))
    QoS = np.zeros((NQs, runtime))
    locked = np.zeros((NQs, 1))  # time stamps when altered

    # set initial, lock in for gen_window
    schedule, locked = F1_schedule(dec_rates,
                                   ach_rates,
                                   H_num,
                                   0,
                                   gen_window,
                                   locked,
                                   schedule)
    change_locked = np.copy(locked)

    for x in range(runtime):

        # Update quality of service tracking
        for y in range(NQs):
            QoS[y, x] = ach_rates[y, 0] - dec_rates[y, 0]

        # Do pair gen from current schedule
        generated = pair_gen(p_gen, schedule)

        # Achieved rate updates:
        if x > 0:
            ach_rates = rate_updates(ach_rates, generated, x)

        # Decide on schedule:
        schedule, locked = F1_schedule(dec_rates,
                                       ach_rates,
                                       H_num,
                                       x,
                                       gen_window,
                                       locked,
                                       schedule)

        # Decide rate changes:
        # Alter 2 ways: 1 - Can't change rate when schedule locked?
        # 2 - can only change after time flux_delay from last change.
        # 2 is implemented
        dec_rates, change_locked = F1_rate_declare(change_rate,
                                                   mean_rate,
                                                   rate_var,
                                                   x,
                                                   flux_delay,
                                                   dec_rates,
                                                   change_locked)
    return QoS


# Output arrays of QoS for plotting and averaged QoS stats
# No change rate: fix a set of rate declarations at beginning
def stripped_simulation(NumUsers: int,
                        H_num: int,
                        p_gen: float,
                        mean_rate: float,
                        rate_var: float,
                        runtime: int) -> list:

    NQs = int(bc(NumUsers, 2))
    schedule = np.zeros((NQs, 1))
    dec_rates = np.ones((NQs, 1)) * (mean_rate - 0.01)
    ach_rates = np.zeros((NQs, 1))
    QoS = np.zeros((NQs, runtime))
    locked = np.zeros((NQs, 1))  # time stamps when altered

    # get gaussian rate declatations
    # use same declarations for entire sim
    dec_rates, fake_locked = F1_rate_declare(1,
                                             mean_rate,
                                             rate_var,
                                             0,
                                             0,
                                             dec_rates,
                                             locked)
    # get new starting schedule, fake time 1
    schedule, fake_locked = F1_schedule(dec_rates,
                                        ach_rates,
                                        H_num,
                                        0,
                                        0,
                                        locked,
                                        schedule)
    # start simulation
    for x in range(runtime):
        # Update quality of service tracking
        for y in range(NQs):
            QoS[y, x] = ach_rates[y, 0] - dec_rates[y, 0]

        # Do pair gen from current schedule
        generated = pair_gen(p_gen, schedule)

        # Achieved rate updates:
        if x > 0:
            ach_rates = rate_updates(ach_rates, generated, x)

            # Decide on schedule:
            schedule, fake_locked = F1_schedule(dec_rates,
                                                ach_rates,
                                                H_num,
                                                x,
                                                0,
                                                locked,
                                                schedule)
    queue_avrgs = np.zeros((NQs, 1))
    for x in range(NQs):
        queue_avrgs[x, 0] = np.sum(QoS[x, :]) / runtime

    Avrg_QoS = np.sum(queue_avrgs[:, 0]) / NQs

    return QoS, queue_avrgs, Avrg_QoS


def plot_QoS(QoS: np.ndarray,
             NumUsers: int,
             H_num: int,
             mean_rate: float,
             rate_var: float,
             p_gen: float,
             time: int) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, int(bc(NumUsers, 2)))
    plt.figure(figsize=(10, 8))
    plt.title('N={}, H={}, mu={}, var={}, p={}'.format(
              NumUsers, H_num, mean_rate, rate_var, p_gen),
              fontsize=28)
    for x in range(int(bc(NumUsers, 2))):
        plt.plot(range(time - int(time / 100)), QoS[x, int(time / 100):],
                 color=cmap(inds[x]), label='{}'.format(x))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(fontsize=20)
    plt.xlabel('Time', fontsize=28)
    plt.ylabel('QoS: achieved - declared rate', fontsize=28)

    timestrp = strftime('%d_%m_%Y_%H_%M', localtime())
    figname = '../Figures/RateFigs/Stripped/{}'.format(timestrp)
    plt.savefig(figname, dpi=300, bbox_inches='tight')


def sample_stripped_QoS(NumUsers: int,
                        H_num: int,
                        p_gen: float,
                        mean_rate: float,
                        rate_var: float,
                        runtime: int,
                        iters: int) -> None:

    AQoS = np.zeros((iters))
    # c95 = [((H_num * p_gen / bc(NumUsers, 2))
    #         - mean_rate) - (1.96 * sqrt(rate_var)),
    #        ((H_num * p_gen / bc(NumUsers, 2))
    #         - mean_rate) + (1.96 * sqrt(rate_var))]
    #
    # c99 = [((H_num * p_gen / bc(NumUsers, 2))
    #         - mean_rate) - (2.58 * sqrt(rate_var)),
    #        ((H_num * p_gen / bc(NumUsers, 2))
    #         - mean_rate) + (2.58 * sqrt(rate_var))]
    #
    # pass95 = 0
    # pass99 = 0
    for x in range(iters):
        QoS, queue_avrgs, single_AQoS = stripped_simulation(NumUsers,
                                                            H_num,
                                                            p_gen,
                                                            mean_rate,
                                                            rate_var,
                                                            runtime)
        AQoS[x] = single_AQoS
        # if (single_AQoS > c95[0]) and (single_AQoS < c95[1]):
        #     pass95 += 1
        # if (single_AQoS > c99[0]) and (single_AQoS < c99[1]):
        #     pass99 += 1

    realized_avrg = np.sum(AQoS) / iters

    print(realized_avrg, '\n\n')

    filename = './TextOut/{}_{}_{}.txt'.format(NumUsers, H_num, runtime)
    afile = open(filename, 'a')
    afile.write('Iters: {} \n'.format(iters))
    afile.write('Mean rate, rate variance, pair gen probability: \
                {}, {}, {}'.format(mean_rate, rate_var, p_gen))
    afile.write('\n All samples average: {} \n\n\n'.format(realized_avrg))
    afile.close()
    # print(c95, pass95, '\n\n')
    # print(c99, pass99)


# time = 10000
# QoS = simulate_F1_service(4, 2, 0.25, 10, 20, 0.05, 0.15, 0.02, time)
# for x in range(int(bc(4, 2))):
#     plt.plot(range(time - int(time / 100)), QoS[x, int(time / 100):])
# plt.show()

# Stripped down simulation
mu = 0.15
var = 0.005
p_gen = 0.25
time = 1000
# QoS, queue_avrgs, AQoS = stripped_simulation(4, 2, p_gen, mu, var, time)
# print('Queue avrgs: ', queue_avrgs, '\n\n', 'Average QoS: ', AQoS)
# plot_QoS(QoS, 4, 2, mu, var, p_gen, time)

sample_stripped_QoS(4, 2, p_gen, mu, var, time, 10000)


# How will continuous time simulation work?
# Run for time x \in [0, Stop)

# At each x check if any pair wants to alter rate declaration
# Decide if rate changes randomly (random number < change rate)

# Rate declare drawn from gaussian dist with mean less than the
# expected single share of the threshold, variance of distance from mean to
# expected single share

# If one or more rate declarations changed, wait time x = tau before
# re-scheduling

# Gen pairs for all scheduled: Bernoulli process with mean p_succ
# Update recorded rate every time a pair successfully created

# Track quality of service by tracking difference in rates per queue
