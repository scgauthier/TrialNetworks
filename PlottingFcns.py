import numpy as np
import matplotlib.pyplot as plt

from scipy.special import binom as bc
from typing import List
# from icecream import ic


def record_flex_dist_fac(timeString: str,
                         params: dict,
                         dist_fac: float) -> None:
    dictName = '../DataOutput/{}'.format(timeString)
    fileName = dictName + '/paramLog.txt'
    # write number of users to param file
    afile = open(fileName, 'a')
    afile.write('Flex dist factor: {}\n'.format(dist_fac))
    afile.close()
    return


def set_dist_fac(timeString: str) -> float:

    # Load param dict from text file
    dictName = '../DataOutput/{}'.format(timeString)
    fileName = dictName + '/paramLog.txt'
    params = {}
    with open(fileName, 'r') as afile:
        # linenum = 0
        for line in afile.readlines():
            # print(line)
            try:
                params['{}'.format(line.split(':')[0])
                       ] = float(line.split(':')[1])
            except ValueError:
                params['{}'.format(line.split(':')[0])
                       ] = line.split(':')[1]

    # Read rates from text file
    fileName = dictName + '/AvReq.txt'
    rates = np.loadtxt(fileName)

    if params['param_change'][1:5] == 'True':

        # Read trk_list from text file
        fileName = dictName + '/trkList.txt'
        trk_list = np.loadtxt(fileName)

        iters, changes = int(params['iters']), int(params['changes'])
        indices = np.linspace(iters/changes, iters, changes)
        spacing = int(indices[1] - indices[0])
        buffDist = int(0.2 * spacing)

        if params['change_key'][1:8] == 'ChangeH':
            H_num, p_gen = int(params['H_num']), params['p_gen']
            # dist_fac = params['dist_fac']
            thresholds = [((H_num * p_gen)
                          // (1/10000)) / 10000]
            for H in trk_list:
                thresholds.append(((H * p_gen)
                                  // (1/10000)) / 10000)

        dist_fac = 0
        for interval in range(1, changes):
            maxPt = max(rates[
                        (interval * spacing) + buffDist:((
                         interval + 1) * spacing) - 1])
            minPt = min(rates[
                        (interval * spacing) + buffDist:((
                         interval + 1) * spacing) - 1])

            maxRelDist = abs((maxPt - thresholds[interval])
                             / thresholds[interval])
            minRelDist = abs((minPt - thresholds[interval])
                             / thresholds[interval])

            if (maxRelDist > dist_fac) or (minRelDist > dist_fac):
                dist_fac = max(maxRelDist, minRelDist)

    else:
        iters = int(params['iters'])
        H_num, p_gen = int(params['H_num']), params['p_gen']
        # dist_fac = params['dist_fac']
        threshold = ((H_num * p_gen)
                     // (1/10000)) / 10000
        buffDist = int(1.0 * (iters / 10))
        dist_fac = 0
        maxPt = max(rates[buffDist:])
        minPt = min(rates[buffDist:])

        maxRelDist = abs((maxPt - threshold)
                         / threshold)
        minRelDist = abs((minPt - threshold)
                         / threshold)

        if (maxRelDist > dist_fac) or (minRelDist > dist_fac):
            dist_fac = max(maxRelDist, minRelDist)

    record_flex_dist_fac(timeString, params, dist_fac)

    return dist_fac


def plot_total_rates(rates: np.ndarray, NumUsers: int, params: dict,
                     trk_list: list, figname: str, multiple: bool) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 4)
    if multiple:
        H_num, p_gen = params['H_num'], params['p_gen']
        threshold = ((H_num * p_gen)
                     // (1/10000)) / 10000  # Truncate at 4th place
        iters, dist_fac = params['iters'], params['dist_fac']
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))
        fig.suptitle('N = {}, H = {}, p = {}, T = {}'.format(
                     NumUsers, H_num, p_gen, threshold),
                     fontsize=28)
        av1 = (sum(rates[0, :]) / iters)
        ax1.plot(range(iters), rates[0, :], color=cmap(0),
                 label='T - {}'.format(dist_fac * threshold))
        ax1.plot(range(iters), [av1] * iters, '--',
                 color=cmap(inds[3]), label='{}'.format(round(av1, 3)))
        av2 = (sum(rates[1, :]) / iters)
        ax2.plot(range(iters), rates[1, :], color=cmap(inds[1]),
                 label='T')
        ax2.plot(range(iters), [av2] * iters, '--',
                 color=cmap(0), label='{}'.format(round(av2, 3)))
        av3 = sum(rates[2, :]) / iters
        ax3.plot(range(iters), rates[2, :], color=cmap(inds[2]),
                 label='T + {}'.format(dist_fac * threshold))
        ax3.plot(range(iters), [av3] * iters, '--',
                 color=cmap(0), label='{}'.format(round(av3, 3)))

        ax3.legend(fontsize=22, framealpha=0.6, loc=2)

        ax2.legend(fontsize=22, framealpha=0.6, loc=2)

        ax1.legend(fontsize=22, framealpha=0.6, loc=2)

    else:
        iters = params['iters']
        plt.figure(figsize=(10, 8))
        plt.plot(range(iters), rates, color=cmap(0),
                 label='Requested rates')
        if params['param_change']:
            if params['change_key'] == 'ChangeH':
                H_num, p_gen = params['H_num'], params['p_gen']
                dist_fac = params['dist_fac']
                thresholds = [((H_num * p_gen)
                              // (1/10000)) / 10000]
                for H in trk_list:
                    thresholds.append(((H * p_gen)
                                      // (1/10000)) / 10000)
                guidelines = []
                upper_error = []
                lower_error = []
                tc = params['changes']
                for tick in range(tc):
                    guidelines += [thresholds[tick]] * int(iters / tc)
                    upper_error += [(1 + dist_fac
                                     ) * thresholds[tick]] * int(iters / tc)
                    lower_error += [(1 - dist_fac
                                     ) * thresholds[tick]] * int(iters / tc)
                plt.plot(range(iters), guidelines, '--',
                         color=cmap(inds[3]), label=r'$\lambda_{Switch}$')
                plt.plot(range(iters), upper_error, '--',
                         color=cmap(inds[2]),
                         label=r'$(1 + \delta)\lambda_{Switch}$')
                plt.plot(range(iters), lower_error, '--',
                         color=cmap(inds[1]),
                         label=r'$(1 - \delta)\lambda_{Switch}$')
                plt.legend(fontsize=22, framealpha=0.6, loc=1)
                plt.ylim(0.8 * min(thresholds), 1.2 * max(thresholds))

        else:
            H_num, p_gen = params['H_num'], params['p_gen']
            threshold = ((H_num * p_gen)
                         // (1/10000)) / 10000  # Truncate at 4th place
            dist_fac = params['dist_fac']
            plt.plot(range(iters), [(1 - dist_fac) * threshold] * iters, '--',
                     color=cmap(inds[1]),
                     label=r'$(1 -  \delta)\lambda_{Switch}$')
            plt.plot(range(iters), [threshold] * iters, '--',
                     color=cmap(inds[3]), label=r'$\lambda_{Switch}$')
            plt.plot(range(iters), [(1 + dist_fac) * threshold] * iters, '--',
                     color=cmap(inds[2]),
                     label=r'$(1 +  \delta)\lambda_{Switch}$')
            plt.legend(fontsize=22, framealpha=0.6, loc=4)
            plt.ylim(0.5 * threshold, 1.5 * threshold)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel('t', fontsize=24)
    plt.ylabel('Sum of rate requests', fontsize=24)

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_TR_from_txt(timeString: str) -> None:

    # Load param dict from text file
    dictName = '../DataOutput/{}'.format(timeString)
    fileName = dictName + '/paramLog.txt'
    params = {}
    with open(fileName, 'r') as afile:
        # linenum = 0
        for line in afile.readlines():
            # print(line)
            try:
                # print(line.split(':')[0])
                params['{}'.format(line.split(':')[0])
                       ] = float(line.split(':')[1])
            except ValueError:
                params['{}'.format(line.split(':')[0])
                       ] = line.split(':')[1]

    # Read rates from text file
    fileName = dictName + '/AvReq.txt'
    rates = np.loadtxt(fileName)

    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 4)

    iters = int(params['iters'])
    plt.figure(figsize=(10, 8))
    plt.plot(range(iters), rates, color=cmap(0),
             label='Requested rates')

    H_num, p_gen = int(params['H_num']), params['p_gen']
    dist_fac = set_dist_fac(timeString)
    thresholds = [((H_num * p_gen)
                  // (1/10000)) / 10000]

    if params['param_change'][1:5] == 'True':
        # Read trk_list from text file
        fileName = dictName + '/trkList.txt'
        trk_list = np.loadtxt(fileName)

        if params['change_key'][1:8] == 'ChangeH':
            for H in trk_list:
                thresholds.append(((H * p_gen)
                                  // (1/10000)) / 10000)
            guidelines = []
            upper_error = []
            lower_error = []
            tc = int(params['changes'])
            for tick in range(tc):
                guidelines += [thresholds[tick]] * int(iters / tc)
                upper_error += [(1 + dist_fac
                                 ) * thresholds[tick]] * int(iters / tc)
                lower_error += [(1 - dist_fac
                                 ) * thresholds[tick]] * int(iters / tc)
            plt.plot(range(iters), guidelines, '--',
                     color=cmap(inds[3]), label=r'$\lambda_{Switch}$')
            plt.plot(range(iters), upper_error, '--',
                     color=cmap(inds[2]),
                     label=r'$(1 + \delta)\lambda_{Switch}$')
            plt.plot(range(iters), lower_error, '--',
                     color=cmap(inds[1]),
                     label=r'$(1 - \delta)\lambda_{Switch}$')
            plt.legend(fontsize=22, framealpha=0.6, loc=1)
            plt.ylim(max(min(thresholds) * (1 - (4 * dist_fac)), 0),
                     max(thresholds) * (1 + (4 * dist_fac)))

    else:
        threshold = thresholds[0]
        plt.plot(range(iters), [(1 - dist_fac) * threshold] * iters, '--',
                 color=cmap(inds[1]),
                 label=r'$(1 -  \delta)\lambda_{Switch}$')
        plt.plot(range(iters), [threshold] * iters, '--',
                 color=cmap(inds[3]), label=r'$\lambda_{Switch}$')
        plt.plot(range(iters), [(1 + dist_fac) * threshold] * iters, '--',
                 color=cmap(inds[2]),
                 label=r'$(1 +  \delta)\lambda_{Switch}$')
        plt.legend(fontsize=22, framealpha=0.6, loc=4)
        plt.ylim(max(min(thresholds) * (1 - (4 * dist_fac)), 0),
                 max(thresholds) * (1 + (4 * dist_fac)))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel('t', fontsize=24)
    plt.ylabel('Sum of rate requests', fontsize=24)

    figName = dictName + '/AvReqRates'
    plt.savefig(figName, dpi=300, bbox_inches='tight')

    return


def plot_rate_profile(all_rates: List[np.ndarray], params: dict,
                      fgnm: str, multiple: bool) -> None:

    NumUsers, H_num = params['NumUsers'], params['H_num']
    p_gen, dist_fac = params['p_gen'], params['dist_fac']
    iters, runs = params['iters'], params['runs']

    threshold = ((H_num * p_gen)
                 // (1/10000)) / 10000  # Truncate at 4th place

    cmap = plt.cm.get_cmap('plasma')
    NQs = int(bc(NumUsers, 2))
    inds = np.linspace(0, 0.95, NQs)
    if multiple:
        numlabs = ['T - {}'.format(
                    (((dist_fac * threshold) // (1/1000)) / 1000)),
                   'T',
                   'T + {}'.format(
                    (((dist_fac * threshold) // (1/1000)) / 1000))]
        wordlabs = ['BelowT', 'AtT', 'AboveT']

        for x in range(3):
            plt.figure(figsize=(10, 8))
            for y in range(NQs):
                plt.plot(range(iters), all_rates[x][y, :], color=cmap(inds[y]))
                plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                plt.title(numlabs[x])

                fgname = '../Figures/AlgAdjust/RateProfile_{}_{}_{}_{}'.format(
                          NumUsers, H_num, runs, wordlabs[x])
                plt.savefig(fgname, dpi=300, bbox_inches='tight')

    else:
        plt.figure(figsize=(10, 8))
        for y in range(NQs):
            plt.plot(range(iters), all_rates[y, :], color=cmap(inds[y]),
                     label='s={}'.format(y))
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        plt.xlabel('t', fontsize=24)
        plt.ylabel('Session request rates', fontsize=24)

        plt.savefig(fgname, dpi=300, bbox_inches='tight')


def plot_RP_from_txt(timeString: str) -> None:

    # Load param dict from text file
    dictName = '../DataOutput/{}'.format(timeString)
    fileName = dictName + '/paramLog.txt'
    params = {}
    with open(fileName, 'r') as afile:
        # linenum = 0
        for line in afile.readlines():
            # print(line)
            try:
                params['{}'.format(line.split(':')[0])
                       ] = float(line.split(':')[1])
            except ValueError:
                params['{}'.format(line.split(':')[0])
                       ] = line.split(':')[1]

    NumUsers, iters = int(params['NumUsers']), int(params['iters'])

    cmap = plt.cm.get_cmap('plasma')
    NQs = int(bc(NumUsers, 2))
    inds = np.linspace(0, 0.95, NQs)

    # Read rates from text file
    fileName = dictName + '/RtProf.txt'
    all_rates = np.loadtxt(fileName).reshape((NQs, iters))

    plt.figure(figsize=(10, 8))
    for y in range(NQs):
        plt.plot(range(iters), all_rates[y, :], color=cmap(inds[y]),
                 label='s={}'.format(y))
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel('t', fontsize=24)
    plt.ylabel('Session request rates', fontsize=24)

    figName = dictName + '/RateProfile'
    plt.savefig(figName, dpi=300, bbox_inches='tight')


def plot_delivery_rates(moving_avrgs: np.ndarray, avrg_delivered: list,
                        figname: str, iters: int, ptsInAvrg: int,
                        multiple: bool) -> None:

    cmap = plt.cm.get_cmap('plasma')
    inds = np.linspace(0, 0.85, 3)
    if multiple:
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 8))

        ax1.plot(range(iters), moving_avrgs[0], color=cmap(0),
                 label='{} pt Avrg'.format(ptsInAvrg))
        ax1.plot(range(iters), [avrg_delivered[0]] * iters, '--',
                 color=cmap(inds[2]),
                 label='Avrg={}'.format(round(avrg_delivered[0], 3)))
        ax2.plot(range(iters), moving_avrgs[1], color=cmap(inds[1]),
                 label='{} pt Avrg'.format(ptsInAvrg))
        ax2.plot(range(iters), [avrg_delivered[1]] * iters, '--',
                 color=cmap(0),
                 label='Avrg={}'.format(round(avrg_delivered[1], 3)))
        ax3.plot(range(iters), moving_avrgs[2], color=cmap(inds[2]),
                 label='{} pt Avrg'.format(ptsInAvrg))
        ax3.plot(range(iters), [avrg_delivered[2]] * iters, '--',
                 color=cmap(0),
                 label='Avrg={}'.format(round(avrg_delivered[2], 3)))

        ax3.legend(fontsize=22, framealpha=0.4, loc=1)

        ax2.legend(fontsize=22, framealpha=0.4, loc=1)

        ax1.legend(fontsize=22, framealpha=0.4, loc=1)

    else:
        plt.figure(figsize=(10, 8))
        plt.plot(range(iters), moving_avrgs[0], color=cmap(0),
                 label='{} pt Avrg'.format(ptsInAvrg))
        plt.plot(range(iters), [avrg_delivered[0]] * iters, '--',
                 color=cmap(inds[2]),
                 label='Avrg={}'.format(round(avrg_delivered[0], 3)))
        plt.legend(fontsize=22, framealpha=0.6, loc=1)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    plt.savefig(figname, dpi=300, bbox_inches='tight')

    return
