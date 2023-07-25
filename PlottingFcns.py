import numpy as np
import matplotlib.pyplot as plt

from scipy.special import binom as bc
from typing import List
from matplotlib import rc
rc('text', usetex=True)
# from icecream import ic


def record_flex_dist_fac(timeString: str,
                         params: dict,
                         dist_fac: float,
                         cPts: int) -> None:
    dictName = '../DataOutput/{}'.format(timeString)
    fileName = dictName + '/paramLog.txt'
    # write number of users to param file
    afile = open(fileName, 'a')
    afile.write('Flex dist factor: {}\n'.format(dist_fac))
    if params['param_change'][1:6] == 'False':
        afile.write('Convergence time: {}\n'.format(cPts[0]))
    else:
        afile.write('Crossing points: {}\n'.format(cPts))
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
    cPts: list = []

    if params['param_change'][1:5] == 'True':

        # Read trk_list from text file
        fileName = dictName + '/trkList.txt'
        trk_list = np.loadtxt(fileName)

        iters, changes = int(params['iters']), int(params['changes'])
        indices = np.linspace(iters/changes, iters, changes)
        spacing = int(indices[1] - indices[0])
        # buffDist = int(0.2 * spacing)
        # buffDist = int(0.01 * spacing)

        if params['change_key'][1:8] == 'ChangeH':
            H_num, p_gen = int(params['H_num']), params['p_gen']
            # dist_fac = params['dist_fac']
            thresholds = [((H_num * p_gen)
                          // (1/10000)) / 10000]
            for H in trk_list:
                thresholds.append(((H * p_gen)
                                  // (1/10000)) / 10000)

            dist_fac: float = 0
            # initial convergence
            crossPt: int = np.where(rates[1: spacing - 1
                                          ] <= thresholds[0])[0][0]
            cPts.append(crossPt + 1)

            maxPt: int = max(rates[1 + crossPt:
                             spacing - 1])
            minPt: int = min(rates[1 + crossPt:
                             spacing - 1])

            maxRelDist = abs((maxPt - thresholds[0])
                             / thresholds[0])
            minRelDist = abs((minPt - thresholds[0])
                             / thresholds[0])

            if (maxRelDist > dist_fac) or (minRelDist > dist_fac):
                dist_fac = max(maxRelDist, minRelDist)

            # follow ups
            for interval in range(1, changes):
                # convergence based on buffDist
                # for interval in range(0, changes):
                # maxPt = max(rates[
                #             (interval * spacing) + buffDist:((
                #              interval + 1) * spacing) - 1])
                # minPt = min(rates[
                #             (interval * spacing) + buffDist:((
                #              interval + 1) * spacing) - 1])

                # convergence based on crossing Pt
                if thresholds[interval] <= thresholds[interval - 1]:
                    crossPt: int = np.where(rates[(interval * spacing) + 1: ((
                                interval + 1) * spacing) - 1
                                ] <= thresholds[interval])[0][0]
                else:
                    try:
                        crossPt: int = np.where(rates[(interval * spacing) + 1: ((
                                    interval + 1) * spacing) - 1
                                    ] >= thresholds[interval])[0][0]
                    except IndexError:
                        crossPt: int = np.where(rates[(interval * spacing) + 1: ((
                                    interval + 1) * spacing) - 1
                                    ] <= thresholds[interval])[0][0]
                    
                if thresholds[interval] != thresholds[interval - 1]:
                    cPts.append(crossPt + (interval * spacing) + 1)

                maxPt: int = max(rates[(interval * spacing) + 1 + crossPt: ((
                            interval + 1) * spacing) - 1])
                minPt: int = min(rates[(interval * spacing) + 1 + crossPt: ((
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
        # Convergence based on looking past first point crossing threshold
        crossPt: int = np.where(rates <= threshold)[0][0]
        cPts.append(crossPt)
        dist_fac = 0
        maxPt = max(rates[crossPt:])
        minPt = min(rates[crossPt:])
        # Convergence based on looking past some buffer point
        # buffDist = int(5.0 * (iters / 10))
        # dist_fac = 0
        # maxPt = max(rates[buffDist:])
        # minPt = min(rates[buffDist:])

        maxRelDist = abs((maxPt - threshold)
                         / threshold)
        minRelDist = abs((minPt - threshold)
                         / threshold)

        if (maxRelDist > dist_fac) or (minRelDist > dist_fac):
            dist_fac = max(maxRelDist, minRelDist)

    record_flex_dist_fac(timeString, params, dist_fac, cPts)

    return dist_fac, cPts


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
                         color=cmap(inds[3]), label=r'$\lambda_{EGS}$')
                plt.plot(range(iters), upper_error, '--',
                         color=cmap(inds[2]),
                         label=r'$(1 + \delta)\lambda_{EGS}$')
                plt.plot(range(iters), lower_error, '--',
                         color=cmap(inds[1]),
                         label=r'$(1 - \delta)\lambda_{EGS}$')
                plt.legend(fontsize=22, framealpha=0.6, loc=1)
                plt.ylim(0.8 * min(thresholds), 1.2 * max(thresholds))

        else:
            H_num, p_gen = params['H_num'], params['p_gen']
            threshold = ((H_num * p_gen)
                         // (1/10000)) / 10000  # Truncate at 4th place
            dist_fac = params['dist_fac']
            plt.plot(range(iters), [(1 - dist_fac) * threshold] * iters, '--',
                     color=cmap(inds[1]),
                     label=r'$(1 -  \delta)\lambda_{EGS}$')
            plt.plot(range(iters), [threshold] * iters, '--',
                     color=cmap(inds[3]), label=r'$\lambda_{EGS}$')
            plt.plot(range(iters), [(1 + dist_fac) * threshold] * iters, '--',
                     color=cmap(inds[2]),
                     label=r'$(1 +  \delta)\lambda_{EGS}$')
            plt.legend(fontsize=22, framealpha=0.6, loc=4)
            plt.ylim(0.5 * threshold, 1.5 * threshold)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel('t', fontsize=24)
    plt.ylabel('Sum of rate requests', fontsize=24)

    plt.savefig(figname, dpi=300, bbox_inches='tight')


def plot_TR_from_txt(timeString: str) -> None:

    axLabelFt = 24
    axTickFt = 22
    legendFt = 20
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

    iters = int(params['iters'])

    H_num, p_gen = int(params['H_num']), params['p_gen']
    dist_fac, cPts = set_dist_fac(timeString)
    thresholds = [((H_num * p_gen)
                  // (1/10000)) / 10000]

    inds = np.linspace(0, 0.85, 4)
    plt.figure(figsize=(14, 8))
    plt.plot(range(iters), rates, color=cmap(0),
             label=r'$\sum_s \lambda_s(t_n)$')

    if params['param_change'][1:5] == 'True':
        # Read trk_list from text file
        fileName = dictName + '/trkList.txt'
        trk_list = np.loadtxt(fileName)

        if params['change_key'][1:8] == 'ChangeH':
            for H in trk_list:
                thresholds.append(((H * p_gen)
                                  // (1/10000)) / 10000)
            lw = 2.0
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
                     color=cmap(inds[3]), linewidth=lw,
                     label=r'$\lambda_{EGS}$')
            plt.plot(range(iters), upper_error, '--',
                     color=cmap(inds[2]), linewidth=lw,
                     label=r'$(1 + \delta)\lambda_{EGS}$')
            plt.plot(range(iters), lower_error, '--',
                     color=cmap(inds[1]), linewidth=lw,
                     label=r'$(1 - \delta)\lambda_{EGS}$')
            ax = plt.gca()
            ax.xaxis.offsetText.set_fontsize(axTickFt)
            for interval in range(1, len(cPts) + 1): 
                plt.axvline(x=cPts[interval - 1],
                            linestyle=':',
                            linewidth=lw, color='k')
            plt.legend(fontsize=legendFt, framealpha=0.6, loc=4)
            plt.ylim(max(min(thresholds) * (1 - (4 * dist_fac)), 0),
                     max(thresholds) * (1 + (4 * dist_fac)))

    else:
        threshold = thresholds[0]
        lw = 2.0
        plt.plot(range(iters), [(1 - dist_fac) * threshold] * iters, '--',
                 color=cmap(inds[1]), linewidth=lw,
                 label=r'$(1 -  \delta)\lambda_{EGS}$')
        plt.plot(range(iters), [threshold] * iters, '--',
                 color=cmap(inds[3]), linewidth=lw,
                 label=r'$\lambda_{EGS}$')
        plt.plot(range(iters), [(1 + dist_fac) * threshold] * iters, '--',
                 color=cmap(inds[2]), linewidth=lw,
                 label=r'$(1 +  \delta)\lambda_{EGS}$')
        plt.legend(fontsize=legendFt, framealpha=0.6, loc=4)
        plt.ylim(max(min(thresholds) * (1 - (4 * dist_fac)), 0),
                 max(thresholds) * (1 + (4 * dist_fac)))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.tick_params(axis='both', labelsize=axTickFt)
    plt.xlabel(r'$t_n$', fontsize=axLabelFt)
    plt.ylabel(r'Rates', fontsize=axLabelFt)
    plt.xlim((0, iters))
    # plt.ylabel('Sum of rate requests', fontsize=24)

    figName = dictName + '/AvReqRates_LongPaper'
    plt.savefig(figName, dpi=300, bbox_inches='tight')

    return


def plot_multiple_from_text(timeString1: str,
                            timeString2: str) -> None:
    # Load param dict from first text file
    dictName = '../DataOutput/{}'.format(timeString1)
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

    # Read rates from first text file
    fileName = dictName + '/AvReq.txt'
    rates1 = np.loadtxt(fileName)

    # Read rates from second text file
    dictName = '../DataOutput/{}'.format(timeString2)
    fileName = dictName + '/AvReq.txt'
    rates2 = np.loadtxt(fileName)

    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')
    iters = int(params['iters'])

    H_num, p_gen = int(params['H_num']), params['p_gen']
    # NumUsers = int(params['NumUsers'])
    dist_fac1 = set_dist_fac(timeString1)
    dist_fac2 = set_dist_fac(timeString2)
    scaler = max(dist_fac1, dist_fac2)
    thresholds = [((H_num * p_gen)
                  // (1/10000)) / 10000]

    inds = np.linspace(0, 0.85, 4)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 14))
    # fig.suptitle('N = {}'.format(
    #              NumUsers),
    #              fontsize=28,
    #              y=0.92)
    ax1.plot(range(iters), rates1, color=cmap(0),
             label='Requested rates')
    ax2.plot(range(iters), rates2, color=cmap(0))

    if params['param_change'][1:5] == 'True':
        # Read trk_list from text file
        fileName = dictName + '/trkList.txt'
        trk_list = np.loadtxt(fileName)

        if params['change_key'][1:8] == 'ChangeH':
            for H in trk_list:
                thresholds.append(((H * p_gen)
                                  // (1/10000)) / 10000)
            guidelines = []
            upper_error1 = []
            lower_error1 = []
            upper_error2 = []
            lower_error2 = []
            tc = int(params['changes'])
            for tick in range(tc):
                guidelines += [thresholds[tick]] * int(iters / tc)
                upper_error1 += [(1 + dist_fac1
                                  ) * thresholds[tick]] * int(iters / tc)
                lower_error1 += [(1 - dist_fac1
                                  ) * thresholds[tick]] * int(iters / tc)
                upper_error2 += [(1 + dist_fac2
                                  ) * thresholds[tick]] * int(iters / tc)
                lower_error2 += [(1 - dist_fac2
                                  ) * thresholds[tick]] * int(iters / tc)
            lw = 2.0
            ax1.plot(range(iters), guidelines, '--',
                     color=cmap(inds[3]), label=r'$\lambda_{EGS}$',
                     linewidth=lw)
            ax1.plot(range(iters), upper_error1, '--',
                     color=cmap(inds[2]),
                     label=r'$(1 + \delta_1)\lambda_{EGS}$',
                     linewidth=lw)
            ax1.plot(range(iters), lower_error1, '--',
                     color=cmap(inds[1]),
                     label=r'$(1 - \delta_1)\lambda_{EGS}$',
                     linewidth=lw)
            ax2.plot(range(iters), guidelines, '--',
                     color=cmap(inds[3]),
                     linewidth=lw)
            ax2.plot(range(iters), upper_error2, '--',
                     color=cmap(inds[2]),
                     label=r'$(1 + \delta_2)\lambda_{EGS}$',
                     linewidth=lw)
            ax2.plot(range(iters), lower_error2, '--',
                     color=cmap(inds[1]),
                     label=r'$(1 - \delta_2)\lambda_{EGS}$',
                     linewidth=lw)
            ax1.legend(fontsize=22, framealpha=0.6, loc=4)
            ax2.legend(fontsize=22, framealpha=0.6, loc=4)
            ax1.set_ylim(max(min(thresholds) * (1 - (2 * scaler)), 0),
                         max(thresholds) * (1 + (2 * scaler)))
            ax2.set_ylim(max(min(thresholds) * (1 - (2 * scaler)), 0),
                         max(thresholds) * (1 + (2 * scaler)))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    plt.xlabel(r'$t_n$', fontsize=24)
    # plt.ylabel('Sum of rate requests',
    #            fontsize=24)
    plt.subplots_adjust(hspace=0.05)

    figName = dictName + '/AvReqRatesMaxandDoubleNonUnf'
    plt.savefig(figName, dpi=300, bbox_inches='tight')

    return


def plot_max_min_diff(timeString1: str, timeString2: str,
                      timeString3: str, timeString4: str,
                      timeString5: str, timeString6: str) -> None:
    # Load param dict from first text file
    dictName = '../DataOutput/{}'.format(timeString1)
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

    # Read max and min rates from first text file
    fileName = dictName + '/MaxReq.txt'
    max1 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min1 = np.loadtxt(fileName)

    # Read rates from second text file
    dictName = '../DataOutput/{}'.format(timeString2)
    fileName = dictName + '/MaxReq.txt'
    max2 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min2 = np.loadtxt(fileName)

    diff1 = max1 - min1
    diff2 = max2 - min2

    # Read rates from third text file
    dictName = '../DataOutput/{}'.format(timeString3)
    fileName = dictName + '/MaxReq.txt'
    max3 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min3 = np.loadtxt(fileName)

    # Read rates from fourth text file
    dictName = '../DataOutput/{}'.format(timeString4)
    fileName = dictName + '/MaxReq.txt'
    max4 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min4 = np.loadtxt(fileName)

    diff3 = max3 - min3
    diff4 = max4 - min4

    # Read rates from fifth text file
    dictName = '../DataOutput/{}'.format(timeString5)
    fileName = dictName + '/MaxReq.txt'
    max5 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min5 = np.loadtxt(fileName)

    # Read rates from sixth text file
    dictName = '../DataOutput/{}'.format(timeString6)
    fileName = dictName + '/MaxReq.txt'
    max6 = np.loadtxt(fileName)
    fileName = dictName + '/minReq.txt'
    min6 = np.loadtxt(fileName)

    diff5 = max5 - min5
    diff6 = max6 - min6

    # max_val1 = max(np.max(diff1[1000:]), np.max(diff2[1000:]))
    # max_val2 = max(np.max(diff3[1000:]), np.max(diff4[1000:]))
    # max_val3 = max(np.max(diff5[1000:]), np.max(diff6[1000:]))

    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')
    iters = int(params['iters'])
    max_iter = int(1e4)
    if max_iter > iters:
        max_iter = iters

    inds = np.linspace(0, 0.85, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(16, 14))

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')

    ax1.plot(range(1, max_iter), diff1[1:max_iter], color=cmap(0))
    ax1.plot(range(1, max_iter), diff2[1:max_iter], color=cmap(inds[1]))

    # ax1.set_ylim((-0.02 * max_val1, 1.1 * max_val1))

    # ax1.legend(fontsize=22, framealpha=0.6, loc=4)

    ax2.plot(range(1, max_iter), diff3[1:max_iter], color=cmap(0))
    ax2.plot(range(1, max_iter), diff4[1:max_iter], color=cmap(inds[1]))

    # ax2.set_ylim((-0.02 * max_val2, 1.1 * max_val2))

    # ax2.legend(fontsize=22, framealpha=0.6, loc=4)

    ax3.plot(range(1, max_iter), diff5[1:max_iter], color=cmap(0),
             label=r'Uniform $\lambda_{u}$')
    ax3.plot(range(1, max_iter), diff6[1:max_iter], color=cmap(inds[1]),
             label=r'Non-uniform $\lambda_u$')

    # ax3.set_ylim((-0.02 * max_val3, 1.1 * max_val3))

    ax3.legend(fontsize=28, framealpha=0.3, loc=1)

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.tick_params(axis='both', labelsize=32)
    ax2.tick_params(axis='both', labelsize=32)
    ax3.tick_params(axis='both', labelsize=32)
    ax3.set_xlim((0, max_iter))
    ax3.xaxis.offsetText.set_fontsize(32)

    plt.xlabel(r'$t_n$', fontsize=34)
    ax2.set_ylabel(r'Average $\max(\lambda_s(t_n)) - $ average $\min(\lambda_s(t_n))$', fontsize=34)

    plt.subplots_adjust(hspace=0.1)

    figName = dictName + '/MaxMinDiffs'
    plt.savefig(figName, dpi=300, bbox_inches='tight')

    return


def plot_convergence_study(timeString1: str, timeString2: str,
                           timeString3: str) -> None:
    
    axLabelFt = 28
    axTickFt = 26
    legendFontSz = 24
    # Load param dict from first text file
    dictName = '../DataOutput/{}'.format(timeString1)
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

    # Read max and min rates from first text file
    fileName = dictName + '/AvReq.txt'
    rates1 = np.loadtxt(fileName)

    # Read rates from second text file
    dictName = '../DataOutput/{}'.format(timeString2)
    fileName = dictName + '/AvReq.txt'
    rates2 = np.loadtxt(fileName)

    # Read rates from third text file
    dictName = '../DataOutput/{}'.format(timeString3)
    fileName = dictName + '/AvReq.txt'
    rates3 = np.loadtxt(fileName)

    max_val = max(np.max(rates1[10000:]), np.max(rates2[10000:]),
                  np.max(rates3[10000:]))
    min_val = min(np.min(rates1[10000:]), np.min(rates2[10000:]),
                  np.min(rates3[10000:]))

    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')
    iters = int(params['iters'])

    H_num, p_gen = int(params['H_num']), params['p_gen']

    dist_fac1, cross1 = set_dist_fac(timeString1)
    dist_fac2, cross2 = set_dist_fac(timeString2)
    dist_fac3, cross3 = set_dist_fac(timeString3)

    cross1, cross2, cross3 = cross1[0], cross2[0], cross3[0]

    threshold = H_num * p_gen
    guidelines = [threshold] * int(iters)
    upper_error1 = [(1 + dist_fac1
                     ) * threshold] * int(iters)
    lower_error1 = [(1 - dist_fac1
                     ) * threshold] * int(iters)
    upper_error2 = [(1 + dist_fac2
                     ) * threshold] * int(iters)
    lower_error2 = [(1 - dist_fac2
                     ) * threshold] * int(iters)
    upper_error3 = [(1 + dist_fac3
                     ) * threshold] * int(iters)
    lower_error3 = [(1 - dist_fac3
                     ) * threshold] * int(iters)


    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')
    iters = int(params['iters'])
    lw = 2.0
    inds = np.linspace(0, 0.85, 4)
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(16, 14))

    ax1.plot(range(iters), rates1, color=cmap(0))
    ax1.plot(range(iters), guidelines,
             '--', linewidth=lw,
             color=cmap(inds[3]))
    ax1.plot(range(iters), upper_error1,
             '--', linewidth=lw,
             color=cmap(inds[2]),
             label=r'$(1 + \delta_1)\lambda_{EGS}$')
    ax1.plot(range(iters), lower_error1,
             '--', linewidth=lw,
             color=cmap(inds[1]),
             label=r'$(1 - \delta_1)\lambda_{EGS}$')
    ax1.axvline(x=cross1, linestyle=':', linewidth=lw, color='k')
    ax1.set_ylim((0.95 * min_val, 1.03 * max_val))

    ax1.legend(fontsize=legendFontSz, framealpha=0.85, loc=3)

    ax2.plot(range(iters), rates2, color=cmap(0))
    ax2.plot(range(iters), guidelines,
             '--', linewidth=lw,
             color=cmap(inds[3]))
    ax2.plot(range(iters), upper_error2,
             '--', linewidth=lw,
             color=cmap(inds[2]),
             label=r'$(1 + \delta_2)\lambda_{EGS}$')
    ax2.plot(range(iters), lower_error2,
             '--', linewidth=lw,
             color=cmap(inds[1]),
             label=r'$(1 - \delta_2)\lambda_{EGS}$')
    ax2.axvline(x=cross2, linestyle=':', linewidth=lw, color='k')

    ax2.set_ylim((0.95 * min_val, 1.03 * max_val))

    ax2.legend(fontsize=legendFontSz, framealpha=0.85, loc=3)

    # ax3.plot(range(iters), rates3, color=cmap(0),
    #          label='Sum demand rates')
    ax3.plot(range(iters), rates3, color=cmap(0),
             label=r'$\sum_s \lambda_s(t_n)$')
    ax3.plot(range(iters), guidelines,
             '--', linewidth=lw,
             label=r'$\lambda_{EGS}$',
             color=cmap(inds[3]))
    ax3.plot(range(iters), upper_error3,
             '--', linewidth=lw,
             color=cmap(inds[2]),
             label=r'$(1 + \delta_3)\lambda_{EGS}$')
    ax3.plot(range(iters), lower_error3,
             '--', linewidth=lw,
             color=cmap(inds[1]),
             label=r'$(1 - \delta_3)\lambda_{EGS}$')
    ax3.axvline(x=cross3, linestyle=':', linewidth=lw, color='k')

    ax3.set_ylim((0.95 * min_val, 1.03 * max_val))
    ax3.set_xlim((0, 25000))
    ax1.set_yticks([0.13, 0.14, 0.15, 0.16, 0.17])
    ax2.set_yticks([0.13, 0.14, 0.15, 0.16, 0.17])
    ax3.set_yticks([0.13, 0.14, 0.15, 0.16, 0.17])

    ax3.legend(fontsize=legendFontSz, framealpha=0.85, loc=3)
    ax3.xaxis.offsetText.set_fontsize(axTickFt)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax1.tick_params(axis='both', labelsize=axTickFt)
    ax2.tick_params(axis='both', labelsize=axTickFt)
    ax3.tick_params(axis='both', labelsize=axTickFt)
    plt.xlabel(r'$t_n$', fontsize=axLabelFt)
    ax2.set_ylabel(r'Rates', fontsize=axLabelFt)
    # ax2.set_ylabel('Sum of requested session rates', fontsize=24)

    plt.subplots_adjust(hspace=0.08)

    figName = dictName + '/ConvergenceStudy_LongPaper'
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


def compare_delivery_from_text(timeString1: str,
                               timeString2: str) -> None:

    dictName1 = '../DataOutput/{}'.format(timeString1)

    # Load param dict from text file
    fileName = dictName1 + '/paramLog.txt'
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

    dictName2 = '../DataOutput/{}'.format(timeString2)

    # Read deliveries from text files
    fileName1 = dictName1 + '/AvDel.txt'
    delivered1 = np.loadtxt(fileName1)
    fileName2 = dictName2 + '/AvDel.txt'
    delivered2 = np.loadtxt(fileName2)

    # Start actual plotting
    cmap = plt.cm.get_cmap('plasma')

    iters = int(params['iters'])

    inds = np.linspace(0, 0.85, 2)
    plt.figure(figsize=(10, 8))
    plt.plot(range(iters), delivered1, color=cmap(0),
             label='RCP')
    plt.plot(range(iters), delivered2, color=cmap(inds[1]),
             label='Fixed')

    min_cand1 = min(delivered1)
    min_cand2 = min(delivered2)
    min_val = min(min_cand1, min_cand2)

    max_cand1 = max(delivered1)
    max_cand2 = max(delivered2)
    max_val = max(max_cand1, max_cand2)

    plt.legend(fontsize=22, framealpha=0.6, loc=1)
    plt.ylim(min_val * (1 - 0.1),
             max_val * (1 + 0.1))

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel('t', fontsize=24)
    plt.ylabel('Entangled Pair Delivery', fontsize=24)

    figName = dictName1 + '/AvDel'
    plt.savefig(figName, dpi=300, bbox_inches='tight')

    return
