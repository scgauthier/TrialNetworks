import numpy as np
from scipy.special import binom as bc
from DistributedNUM import sim_QL_w_rate_feedback

# The purpose of these functions is to provide characterization of the
# distributed algorithm for rate adjustment. Characterization focuses on the
# stepsize and how it affects convergence


#############################################################################
#  Write simulation to optimize the stepsize w/o scaling
#  want to record max fluctuations detected after first x steps have passed,
#  for x = 10, 100, 1000; want average queue backlog (discard 1000 steps)
# loop each simulation 100 times, determine average fluctuation size
# fix all investigations to be 10 000 iterations
# default generation probability to be 0.05
# fix central scaling to be by 1/lambda_switch for quick convergence to
# correct backlog.
# fix session min rates to be gen_prob / 1000
# fix user max rates to be ()(NQs - 1) / 2) * p_gen
def characterize_stepsize(NumUsers: int, H_num: int,
                          max_sched_per_q: int) -> None:

    NQs = int(bc(NumUsers, 2))
    p_gen = 0.5
    threshold = ((H_num * p_gen)
                 // (1/10000)) / 10000  # Truncate at 4th place)
    iters = 10000
    global_scale = 1000
    reps_for_avrg = 100
    session_min_rates = [p_gen / global_scale] * NQs
    user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers

    exclusions = [10, 100, 1000]

    filename = '../DataOutput/Char_Steps_{}_{}_{}.txt'.format(
                NumUsers, H_num, max_sched_per_q)
    afile = open(filename, 'a')
    afile.write('Threshold: {} \n\n'.format(threshold))
    afile.close()

    # investigate range of vals, from prop to lambda_switch to 1/lambda_switch
    # step_sizes = [(threshold / 100), (threshold / 10), (threshold / 5),
    #               threshold, 1, (1 / (threshold * 5)), (1 / threshold)]
    # step_sizes = [(2 / threshold), (10 / threshold)]
    step_sizes = [(20 / threshold), (100 / threshold)]

    for step in step_sizes:
        max_detected = np.zeros((3, reps_for_avrg))
        min_detected = np.zeros((3, reps_for_avrg))
        for rep in range(reps_for_avrg):

            print(rep, '\n')

            (queues,
             rts,
             deliveries) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                                  threshold,
                                                  user_max_rates,
                                                  session_min_rates,
                                                  step, p_gen,
                                                  max_sched_per_q,
                                                  iters)
            for x in range(3):
                sum_rates = np.zeros((iters - exclusions[x]))
                for y in range(exclusions[x], iters):
                    sum_rates[y - exclusions[x]] = np.sum(rts[:, y], axis=0)

                max_detected[x, rep] = max(sum_rates)
                min_detected[x, rep] = min(sum_rates)

        for x in range(3):
            av_max = np.sum(max_detected[x, :]) / reps_for_avrg
            av_min = np.sum(min_detected[x, :]) / reps_for_avrg

            afile = open(filename, 'a')
            afile.write('Step size: {}, Points Excluded: {} \n'.format(
                        step, exclusions[x]))
            afile.write('max rate: {}, min rate: {} \n'.format(
                        av_max, av_min))
            afile.write('Fluctuation Size: {} \n\n'.format(
                        av_max - av_min))
            afile.close()

    return


#  Write simulation to optimize the global scaling
#  want to record max fluctuations detected after first x steps have passed,
#  for x = 10, 100, 1000; want average queue backlog (discard 1000 steps)
# loop each simulation 100 times, determine average fluctuation size
# fix all investigations to be 10 000 iterations
# fix step size to be by 1/lambda_switch, as suggested by other simulations
# fix session min rates to be gen_prob / 1000
# fix user max rates to be ((NQs - 1) / 2) * p_gen
def characterize_global_scaling(NumUsers: int, H_num: int,
                                max_sched_per_q: int) -> None:

    NQs = int(bc(NumUsers, 2))
    p_gen = 0.5
    threshold = ((H_num * p_gen)
                 // (1/10000)) / 10000  # Truncate at 4th place)
    iters = 10000
    global_scale = 1000
    reps_for_avrg = 100
    session_min_rates = [p_gen / global_scale] * NQs
    user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers

    exclusions = [10, 100, 1000]

    filename = '../DataOutput/Char_GS_{}_{}_{}.txt'.format(
                NumUsers, H_num, max_sched_per_q)
    afile = open(filename, 'a')
    afile.write('Threshold: {} \n\n'.format(threshold))
    afile.close()

    # investigate range of vals, from 1 to 1/lambda_switch
    scalings = np.linspace(1, 1/threshold, 5)

    for scale in scalings:
        max_detected = np.zeros((3, reps_for_avrg))
        min_detected = np.zeros((3, reps_for_avrg))
        for rep in range(reps_for_avrg):

            print(rep, '\n')

            (queues,
             rts,
             deliveries) = sim_QL_w_rate_feedback(NumUsers, H_num,
                                                  threshold,
                                                  user_max_rates,
                                                  session_min_rates,
                                                  1/threshold,
                                                  scale, p_gen,
                                                  max_sched_per_q,
                                                  iters)

            for x in range(3):
                sum_rates = np.zeros((iters - exclusions[x]))
                for y in range(exclusions[x], iters):
                    sum_rates[y - exclusions[x]] = np.sum(rts[:, y], axis=0)

                max_detected[x, rep] = max(sum_rates)
                min_detected[x, rep] = min(sum_rates)

        for x in range(3):
            av_max = np.sum(max_detected[x, :]) / reps_for_avrg
            av_min = np.sum(min_detected[x, :]) / reps_for_avrg

            afile = open(filename, 'a')
            afile.write('Scaling: {}, Points Excluded: {} \n'.format(
                        scale, exclusions[x]))
            afile.write('max rate: {}, min rate: {} \n'.format(
                        av_max, av_min))
            afile.write('Fluctuation Size: {} \n\n'.format(
                        av_max - av_min))
            afile.close()

    return
