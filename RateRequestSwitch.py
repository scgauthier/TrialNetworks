import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom as bc
from scipy.stats import norm, bernoulli
from random import random, sample


# Parameters:
# -change rate: stipulates how often a single queue changes it's declared rate
# mean_rate: the average rate declared by any queue
# rate_var: variance of declared rates
def F1_rate_declare(change_rate: float,
                    mean_rate: float,
                    rate_var: float,
                    rates: np.ndarray) -> np.ndarray:

    NQs = np.shape(rates)[0]
    up_rates = np.copy(rates)
    for x in range(NQs):
        if random() < (change_rate / NQs):  # Change rate equally distributed
            up_rates[x, 0] = norm.rvs(mean_rate, rate_var, size=1)[0]
    return up_rates


def calc_F1_schedule(declared_rates: np.ndarray,
                     achieved_rates: np.ndarray,
                     H_num: int,
                     time: int,
                     delay: int,
                     locked: np.ndarray,
                     schedule: np.ndarray) -> list:

    NQs = np.shape(declared_rates)[0]
    new_schedule = np.copy(schedule)

    if time > 0:
        alterable = []
        # Determine H effective
        for x in np.nonzero(schedule[:, 0])[0]:
            if (time >= locked[x, 0] + delay):
                alterable.append(x)

        # Leave all unalterable scheduled
        for x in alterable:
            new_schedule[x, 0] = 0

        H_eff = len(alterable)

        if H_eff > 0:
            # Determine rate diffs for schedulable queues
            diffs = []
            for x in range(NQs):
                if x not in np.nonzero(new_schedule[:, 0])[0]:
                    diffs.append(declared_rates[x, 0]
                                 - achieved_rates[x, 0])

            arranged_diffs = sorted(diffs)
            arranged_diffs.reverse()

            # Fill in schedule gaps
            if (np.max(arranged_diffs) != np.min(arranged_diffs)):
                for x in range(H_eff):
                    ind = diffs.index(arranged_diffs[x])
                    new_schedule[ind, 0] = 1
                    locked[ind, 0] = time

                # If no differences, break ties randomly
            else:
                inds = sample(np.nonzero(new_schedule[:, 0])[0], H_eff)
                for x in inds:
                    new_schedule[x, 0] = 1

    # At time 0, assign random schedule
    else:
        inds = sample(range(NQs), H_num)
        for x in inds:
            new_schedule[x, 0] = 1

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

    # set initial
    schedule, locked = calc_F1_schedule(dec_rates,
                                        ach_rates,
                                        H_num,
                                        0,
                                        gen_window,
                                        locked,
                                        schedule)

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
        schedule, locked = calc_F1_schedule(dec_rates,
                                            ach_rates,
                                            H_num,
                                            0,
                                            gen_window,
                                            locked,
                                            schedule)

        # Decide rate changes:
        dec_rates = F1_rate_declare(change_rate,
                                    mean_rate,
                                    rate_var,
                                    dec_rates)
    return QoS


time = 1000
QoS = simulate_F1_service(4, 2, 0.2, 10, 0.05, 0.2, 0.05, time)
for x in range(int(bc(4, 2))):
    plt.plot(range(time - int(time / 100)), QoS[x, int(time / 100):])
plt.show()

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
