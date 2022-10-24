import os
import time
import numpy as np
from scipy.special import binom as bc
from RayDistributedNUM import record_NumUsers, study_algorithm


def load_params(NumUsers: int) -> dict:

    iters = 1000
    runs = 100
    dist_fac = 0.02

    Nexcl = 0

    H_num = 3
    p_gen = 0.05
    global_scale = 1000
    max_sched_per_q = 1
    lambda_Switch = H_num * p_gen
    NQs = int(bc(NumUsers, 2))

    # Should relate to timescale of system
    # One node can be involved in N-1 sessions
    # per session a mx of p_gen ent generated per slot
    # maybe a user can deal with max of ((NQs - 1) / 2) * p_gen pair generated
    # per slot, as example where user cutoffs are actually relevant
    user_max_rates = [((NQs - 1) / 2) * p_gen] * NumUsers
    # user_max_rates = [lambda_Switch] * NumUsers
    # try user_max_rates set to NQs for case when they are not relevant
    # user_max_rates = [NQs] * NQs

    session_min_rates = [p_gen / global_scale] * NQs
    step_size = 1
    central_scale = 1 / lambda_Switch

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
        'central_scale': central_scale,
        'param_change': param_change,
        'change_key': change_key,
        'indices': indices,
        'iters': iters,
        'runs': runs,
        'dist_fac': dist_fac,
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


NumUsers = 4
params = load_params(NumUsers)
record_NumUsers(NumUsers, params)

# study_balance_near_threshold(NumUsers, H_num, user_max_rates,
#                              session_min_rates, step_size,
#                              p_gen, max_sched_per_q,
#                              iters, dist_fac)
study_algorithm(NumUsers, params)
