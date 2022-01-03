from itertools import permutations, combinations
from scipy.special import binom as bc
from math import factorial


def gen_unique_schedules(NumUsers: int,
                         H_num: int) -> list:

    # Step 1: determine total number of feasible schedules
    NumSchedules = (1 / factorial(H_num))
    for m in range(H_num):
        NumSchedules *= bc(NumUsers - (2 * m), 2)

    # Step 2: generate all orderings of nodes in sets of of size 2*H
    nl = range(NumUsers)
    H_permutations = list(permutations(nl, 2 * H_num))

    # Look pairwise at the elements of node sets.
    # Discard elements with unordered pairs
    marked = []
    for scramble in H_permutations:
        for x in range(H_num):
            if scramble[2 * x] > scramble[2 * x + 1]:
                marked.append(scramble)
                break
    for unordered in marked:
        H_permutations.remove(unordered)

    # Gen VOQ list to have reference for queue indices of pairings
    VOQs = list(combinations(nl, 2))

    # Look at sets, determine queue indices of pairs, add to temporary schedule
    # Sort temp schedule. If temp schedule not in schedules, add it.
    # Once total number of schedules found, stop looking.
    schedules = []
    for scramble in H_permutations:
        schedule = []
        for x in range(H_num):
            ind = VOQs.index((scramble[2 * x], scramble[2 * x + 1]))
            schedule.append(ind)
        schedule.sort()
        try:
            schedules.index(schedule)
        except ValueError:
            schedules.append(schedule)

        if len(schedules) == NumSchedules:
            break
    return schedules
