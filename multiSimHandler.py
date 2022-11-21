import os
import time
import numpy as np
from math import log10 as lg
# from math import floor
from scipy.special import binom as bc
from MultiProcDistributedNUM import record_NumUsers, study_algorithm
from MultiProcDistributedNUM import load_user_max_rates


def round_down(NumUsers: int) -> int:
    if NumUsers < 10:
        return 1
    else:
        for x in range(6):
            rDown = round((NumUsers - x), -1)
            if rDown <= NumUsers:
                return rDown


def load_params(NumUsers: int) -> dict:

    iters = 100000
    runs = 1000

    Nexcl = 0

    H_num = 3
    p_gen = 0.05
    global_scale = 1000
    max_sched_per_q = 1
    lambda_Switch = H_num * p_gen
    NQs = int(bc(NumUsers, 2))

    # possible keywords:
    # 1. uniformVeryHigh
    # 2. uniformSessionMax
    # 3. singleNonUniformSessionMax
    # 4. doubleNonUniformSessionMax
    user_max_rates = load_user_max_rates(NumUsers,
                                         p_gen, NQs,
                                         max_sched_per_q,
                                         'uniformVeryHigh')
    session_min_rates = [p_gen / global_scale] * NQs
    # step_size = round_down(NumUsers) / (1 + lg(NumUsers))
    step_size = 1
    # Scale increase slightly with Number of users
    user_scale_factor = lg(NumUsers)
    # user_scale_factor = lg(NumUsers)
    central_scale = user_scale_factor / lambda_Switch

    param_change = True

    # Possible change keys:
    # False, 'ChangeH'
    change_key = 'ChangeH'
    changes = 10
    indices = np.linspace(iters/changes, iters, changes)

    params = {
        'H_num': H_num,
        'p_gen': p_gen,
        'max_sched_per_q': max_sched_per_q,
        'user_max_rates': user_max_rates,
        'session_min_rates': session_min_rates,
        'step_size': step_size,
        'user_scale_factor': user_scale_factor,
        'central_scale': central_scale,
        'param_change': param_change,
        'change_key': change_key,
        'indices': indices,
        'iters': iters,
        'runs': runs,
        'Nexcl': Nexcl,
        'changes': changes,
        'timeStr': time.strftime("%Y%m%d-%H%M%S")
    }

    dirName = '../DataOutput/{}'.format(params['timeStr'])
    fileName = dirName + '/paramLog.txt'

    if not os.path.isdir(dirName):
        os.mkdir(dirName)
    if not os.path.isfile(fileName):

        afile = open(fileName, 'w')

        for key, value in params.items():
            afile.write('{}: {}\n'.format(key, value))
        afile.close()

    return params


if __name__ == '__main__':
    NumUsers = 20
    params = load_params(NumUsers)
    record_NumUsers(NumUsers, params)

    # study_balance_near_threshold(NumUsers, H_num, user_max_rates,
    #                              session_min_rates, step_size,
    #                              p_gen, max_sched_per_q,
    #                              iters)
    study_algorithm(NumUsers, params)
