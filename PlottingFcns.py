import numpy as np
import matplotlib.pyplot as plt

from scipy.special import binom as bc
from typing import List


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


def plot_rate_profile(all_rates: List[np.ndarray], NumUsers: int,
                      H_num: int, p_gen: float, threshold: float,
                      dist_fac: float, iters: int, runs: int,
                      multiple: bool, tag: str) -> None:

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

            fgname = '../Figures/AlgAdjust/RateProfile_{}_{}_{}_{}'.format(
                      NumUsers, H_num, runs, tag)
            plt.savefig(fgname, dpi=300, bbox_inches='tight')


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
