import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import norm, bernoulli
from random import random
from math import sqrt
from time import localtime, strftime
from matplotlib import rc

rc('text', usetex=True)
rc('xtick', labelsize=28)
rc('ytick', labelsize=28)


# Parameters:
# -change rate: stipulates probability a single (unlocked) queue changes
#               it's declared rate
# mean_rate: the average rate declared by any queue
# rate_var: variance of declared rates
# delay: minimum amount of time that must pass b/w updating rate declaration
# change_locked: array of most recent times each rate updated
def rate_declare(change_rate: float,
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
        if (time >= (change_locked[x] + delay)):
            alterable.append(x)
    for x in alterable:
        # if random() < (change_rate / NQs):  # Change rate equally distributed
        if random() < change_rate:
            draw = -1
            while draw < 0:
                draw = norm.rvs(mean_rate, rate_std, size=1)[0]
            up_rates[x] = draw
            change_locked[x] = time
    return up_rates, change_locked


def set_schedule(declared_rates: np.ndarray,
                 achieved_rates: np.ndarray,
                 H_num: int,
                 time: int,
                 delay: int,
                 locked: np.ndarray,
                 schedule: np.ndarray) -> list:

    NQs = np.shape(declared_rates)[0]
    new_schedule = np.copy(schedule)

    # if nothing achieved yet, schedule highest rates
    if np.sum(achieved_rates) == 0:
        inds = np.argpartition(declared_rates, -H_num)[-H_num:]
        for x in range(H_num):
            new_schedule[inds[x]] = 1
            locked[inds[x]] = time

    else:

        # should be +ve if not over achieving
        diffs = list((declared_rates - achieved_rates))

        # keep scheduled if within gen_window time:
        fixed = NQs
        for x in range(NQs):
            if time >= locked[x] + delay:
                new_schedule[x] = 0
                fixed -= 1
            else:
                diffs[x] = -1
        H_eff = H_num - fixed

        arranged_diffs = sorted(diffs)
        arranged_diffs.reverse()

        for x in range(H_eff):

            # Not over blocking
            ind = diffs.index(arranged_diffs[x])
            new_schedule[ind] = 1
            locked[ind] = time

    return new_schedule, locked


# Can add blocking feature to model inability to gen at back to back times
def pair_gen(p_gen: float,
             schedule: np.ndarray
             ) -> np.ndarray:

    NQs = np.shape(schedule)[0]
    generated = np.zeros((NQs))

    for x in np.nonzero(schedule[:])[0]:
        generated[x] = bernoulli.rvs(p_gen, size=1)[0]

    return generated


# For all times greater than 0
def rate_updates(ach_rates: np.ndarray,
                 generated: np.ndarray,
                 time: int) -> np.ndarray:
    for x in range(np.shape(generated)[0]):
        ach_rates[x] = ((ach_rates[x] * (time - 1))
                        + generated[x]) / time  # New rate

    return ach_rates


# All rates start at just less than mean_rate
# gen_window: min length of time stays scheduled for
# flux_delay: min length of time b/w changing rate declarations
# change_rate: probability a node pair changes rate declaration, when able
def simulate_service(NumUsers: int,
                     H_num: int,
                     p_gen: float,
                     gen_window: int,
                     flux_delay: int,
                     change_rate: float,
                     mean_rate: float,
                     rate_var: float,
                     runtime: int) -> np.ndarray:

    NQs = int(bc(NumUsers, 2))
    schedule = np.zeros((NQs))
    dec_rates = np.ones((NQs)) * (0.99 * mean_rate)
    ach_rates = np.zeros((NQs))
    QoS = np.zeros((NQs, runtime))
    locked = np.zeros((NQs))  # time stamps when altered

    # set initial, lock in for gen_window
    schedule, locked = set_schedule(dec_rates,
                                    ach_rates,
                                    H_num,
                                    0,
                                    gen_window,
                                    locked,
                                    schedule)
    change_locked = np.copy(locked)

    for x in range(runtime):

        # Update quality of service tracking
        # Make relative to size of declared rate
        for y in range(NQs):
            QoS[y, x] = (ach_rates[y] - dec_rates[y])

        # Do pair gen from current schedule
        generated = pair_gen(p_gen, schedule)

        # Achieved rate updates:
        if x > 0:
            ach_rates = rate_updates(ach_rates, generated, x)

        # Decide on schedule:
        schedule, locked = set_schedule(dec_rates,
                                        ach_rates,
                                        H_num,
                                        x,
                                        gen_window,
                                        locked,
                                        schedule)

        # rate changes:
        # can only change after time flux_delay from last change.
        dec_rates, change_locked = rate_declare(change_rate,
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
    schedule = np.zeros((NQs))
    dec_rates = np.ones((NQs)) * (0.99 * mean_rate)
    ach_rates = np.zeros((NQs))
    QoS = np.zeros((NQs, runtime))
    locked = np.zeros((NQs))  # time stamps when altered

    # get gaussian rate declatations
    # use same declarations for entire sim
    dec_rates, fake_locked = rate_declare(1,
                                          mean_rate,
                                          rate_var,
                                          0,
                                          0,
                                          dec_rates,
                                          locked)
    print('Requested: ', dec_rates, '\n')

    service_expectations = np.copy(ach_rates)
    for x in range(NQs):
        service_expectations[x] = ((H_num * p_gen
                                   * (dec_rates[x] / np.sum(dec_rates)))
                                   - dec_rates[x])

    print('Service Expectations: ', service_expectations, '\n')
    print('Expected AQoS: ', ((H_num * p_gen) - np.sum(dec_rates)) / NQs)

    # get new starting schedule, fake time 1
    schedule, fake_locked = set_schedule(dec_rates,
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
            QoS[y, x] = ach_rates[y] - dec_rates[y]

        # Do pair gen from current schedule
        generated = pair_gen(p_gen, schedule)

        # Achieved rate updates:
        if x > 0:
            ach_rates = rate_updates(ach_rates, generated, x)

            # Decide on schedule:
            schedule, fake_locked = set_schedule(dec_rates,
                                                 ach_rates,
                                                 H_num,
                                                 x,
                                                 0,
                                                 locked,
                                                 schedule)
    queue_avrgs = np.zeros((NQs))
    for x in range(NQs):
        queue_avrgs[x] = np.sum(QoS[x, :]) / runtime

    Avrg_QoS = np.sum(queue_avrgs) / NQs

    return QoS, queue_avrgs, Avrg_QoS


def plot_QoS(QoS: np.ndarray,
             NumUsers: int,
             H_num: int,
             mean_rate: float,
             rate_var: float,
             p_gen: float,
             time: int) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.9, int(bc(NumUsers, 2)))
    plt.figure(figsize=(10, 8))
    # plt.title('N={}, H={}, mu={}, var={}, p={}'.format(
    # NumUsers, H_num, mean_rate, rate_var, p_gen),
    # fontsize=28)
    for x in range(int(bc(NumUsers, 2))):
        plt.plot(range(time - int(time / 100)), QoS[x, int(time / 100):],
                 color=cmap(inds[x]), label='{}'.format(x))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.legend(loc=2, fontsize=20)
    plt.xlabel('Time', fontsize=28)
    plt.ylabel('QoS: achieved - declared rate', fontsize=28)

    timestrp = strftime('%d_%m_%Y_%H_%M', localtime())
    figname = '../Figures/RateFigs/Stripped/{}'.format(timestrp)
    # plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()


def sample_stripped_QoS(NumUsers: int,
                        H_num: int,
                        p_gen: float,
                        mean_rate: float,
                        rate_var: float,
                        runtime: int,
                        iters: int) -> None:

    AQoS = np.zeros((iters))

    for x in range(iters):
        QoS, queue_avrgs, single_AQoS = stripped_simulation(NumUsers,
                                                            H_num,
                                                            p_gen,
                                                            mean_rate,
                                                            rate_var,
                                                            runtime)
        AQoS[x] = single_AQoS

    realized_avrg = np.sum(AQoS) / iters
    variance = np.var(AQoS)

    print(realized_avrg, '\n\n')

    filename = './TextOut/{}_{}_{}.txt'.format(NumUsers, H_num, runtime)
    afile = open(filename, 'a')
    afile.write('Iters: {} \n'.format(iters))
    afile.write('Mean rate, rate variance, pair gen probability: \
                {}, {}, {}'.format(mean_rate, rate_var, p_gen))
    afile.write('\n All samples average: {} \n'.format(realized_avrg))
    afile.write('\n All samples variance: {} \n\n\n'.format(variance))
    afile.close()


# time = 10000
# QoS = simulate_service(4, 2, 0.25, 10, 20, 0.05, 0.15, 0.02, time)
# for x in range(int(bc(4, 2))):
#     plt.plot(range(time - int(time / 100)), QoS[x, int(time / 100):])
# plt.show()

# Stripped down simulation
p_gen = 2 * 0.05 * 4e-4
mu = 0.85 * p_gen * (1/3)
var = 0.02 * mu
# avrg_time = 2e5
time = int(1e6)

change_rate = 0.001
flux_delay = 100
gen_window = 100
QoS, queue_avrgs, AQoS = stripped_simulation(4, 2, p_gen, mu, var, time)
print('Queue avrgs: ', queue_avrgs, '\n\n', 'Average QoS: ', AQoS)
# QoS = simulate_service(4, 2, p_gen, gen_window, flux_delay, change_rate,
#                        mu, var, time)
plot_QoS(QoS, 4, 2, mu, var, p_gen, time)

# sample_stripped_QoS(4, 2, p_gen, mu, var, time, 10000)
# QoS, queue_avrgs, single_AQoS = stripped_simulation(4,
#                                                     2,
#                                                     p_gen,
#                                                     mu,
#                                                     var,
#                                                     time)
